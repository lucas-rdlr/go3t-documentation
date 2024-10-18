import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SuperGATConv, BatchNorm, TransformerConv
from flax import linen as nn_flax
import jax.numpy as jnp


# Define DefaultConfig for the Wormhole model
class DefaultConfig:
    dtype = jnp.float32
    dist_func_enc = 'GW'
    dist_func_dec = 'GW'
    eps_enc = 0.1
    eps_dec = 0.01
    lse_enc = True
    lse_dec = True
    coeff_dec = 1
    scale = 'min_max_total'
    factor = 1.0
    emb_dim = 64
    num_heads = 2
    num_layers = 2
    qkv_dim = 64
    mlp_dim = 64
    attention_dropout_rate = 0.5
    kernel_init = nn_flax.initializers.glorot_uniform()
    bias_init = nn_flax.initializers.zeros


class TIDE(nn.Module):
    def __init__(self, num_features, time_parameter_initial=1.0):
        super(TIDE, self).__init__()

        self.time_parameter = nn.Parameter(torch.tensor([time_parameter_initial]))
        self.num_features = num_features


    def forward(self, x, edge_index, edge_weight):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        time = torch.sigmoid(self.time_parameter)
        v = edge_weight * time
        size = x.size(0)
        adj = torch.sparse_coo_tensor(edge_index, v, torch.Size([size, size]))
        diffusion = torch.sparse.mm(adj, x)

        return x + diffusion


class Expert(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(Expert, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.5)
        self.bn = BatchNorm(out_channels)
        self.fc = nn.Linear(out_channels, 50)


    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.bn(x)
        x = self.fc(x)

        return x


class SimpleMoE(nn.Module):
    def __init__(self, num_experts, in_channels, hidden_channels, out_channels, heads):
        super(SimpleMoE, self).__init__()

        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(in_channels, hidden_channels, out_channels, heads) for _ in range(num_experts // 2)])
        self.experts.extend([GraphTransformer(in_channels, hidden_channels, out_channels, heads) for _ in range(num_experts // 2)])
        self.gate = nn.Linear(in_channels, num_experts)


    def forward(self, x, edge_index, batch):
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # Compute gate scores
        expert_outputs = [expert(x, edge_index) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_scores = gate_scores.unsqueeze(2).expand_as(expert_outputs)
        output = torch.sum(expert_outputs * gate_scores, dim=1)
        
        return output


class GNNMoE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_experts):
        super(GNNMoE, self).__init__()

        self.tide = TIDE(in_channels)
        self.moe = SimpleMoE(num_experts, in_channels, hidden_channels, out_channels, heads)


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.tide(x, edge_index, edge_weight)
        output = self.moe(x, edge_index, data.batch)

        return output


class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GraphAutoencoder, self).__init__()
        
        self.encoder = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.decoder = nn.Linear(hidden_channels * heads, out_channels)


    def forward(self, x, edge_index):
        encoded = torch.relu(self.encoder(x, edge_index))
        decoded = self.decoder(encoded)

        return encoded, decoded


# Neural Optimal Transport Autoencoder (NOTAE)
class NOTAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NOTAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GraphTransformer, self).__init__()
        self.transformer1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.transformer2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.5)
        self.bn = BatchNorm(out_channels)
        self.fc = nn.Linear(out_channels, 50)
        self.skip_linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        residual = self.skip_linear(x)  # Linear transformation for skip connection
        x = torch.relu(self.transformer1(x, edge_index))
        x = self.transformer2(x, edge_index)
        x = x + residual  # Adding skip connection
        x = torch.relu(x)
        x = self.bn(x)
        x = self.fc(x)
        return x


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden: int, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim


        layers.append(nn.Linear(in_dim, hidden_dim))


        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_low: float,
        features_high: float,
        dim_hidden_encoder: int,
        num_hidden_encoder: int,
        dim_hidden_head: int,
        num_hidden_head: int,
        corruption_rate: float = 0.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()


        self.encoder = MLP(input_dim, dim_hidden_encoder, num_hidden_encoder, dropout)
        self.pretraining_head = MLP(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout)


        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.tensor(features_low, dtype=torch.float32), torch.tensor(features_high, dtype=torch.float32))
        self.corruption_rate = corruption_rate


    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_features = x.size()


        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true
        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size, num_features))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x, x_random)


        # get embeddings
        embeddings = self.pretraining_head(self.encoder(x))
        embeddings_corrupted = self.pretraining_head(self.encoder(x_corrupted))


        return embeddings, embeddings_corrupted


    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class NTXent(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation


        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature


    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch


        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples


        Returns:
            float: loss
        """
        batch_size = z_i.size(0)


        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)


        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)


        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)


        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)


        return loss
    

# Custom Noise Contrastive Loss using Euclidean Distance
class NoiseContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(NoiseContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        similarity = torch.cdist(z, z, p=2)  # Euclidean distance
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(-positives / self.temperature)
        denominator = mask * torch.exp(-similarity / self.temperature)
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        
        return loss
