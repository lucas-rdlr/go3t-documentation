import torch
import torch.nn as nn
import pickle

from torch import GradScaler, autocast
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from src.classes import NTXent, NoiseContrastiveLoss

def train(EPOCHS, LOADER, MODEL, OPTIMIZER, AUTOENCODER, DEVICE, DATASET_ID, NAME, FOLDER):

    # Initialize best_loss to a very high number
    loss_history = []
    best_loss = float('inf')

    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    noise_contrastive_loss = NoiseContrastiveLoss()
    ntxent_loss = NTXent()
    scaler = GradScaler("cuda")

    # Training loop
    for epoch in range(EPOCHS):
        for data in LOADER:
            data = data.to(DEVICE)
            OPTIMIZER.zero_grad()

        # Execute tide operation outside of autocast
        x = MODEL.tide(data.x, data.edge_index, data.edge_weight)

        # Start mixed precision context
        with autocast("cuda"):
            output = MODEL.moe(x, data.edge_index, data.batch)
            mse = mse_loss(output, data.x)
            kl = kl_loss(torch.log_softmax(output, dim=1), torch.softmax(data.x, dim=1))

            # Neural Optimal Transport Autoencoder
            encoded, decoded = AUTOENCODER(data.x)
            ae_loss = mse_loss(decoded, data.x)

            # # Graph autoencoder
            # encoded, decoded = AUTOENCODER(data.x, data.edge_index)
            # ae_loss = mse_loss(decoded, data.x)

            # SCARF model
            # emb_anchor, emb_positive = SCARF_MODEL(data.x)
            # scarf_loss = ntxent_loss(emb_anchor, emb_positive)

            # Noise Contrastive Loss
            noise_encoded = encoded + torch.randn_like(encoded) * 0.1
            contrastive_loss = noise_contrastive_loss(encoded, noise_encoded)

            # Clustering-aware loss
            predicted_clusters = output.argmax(dim=1).cpu().numpy()
            ground_truth = data.y.cpu().numpy()
            ari_loss = 1.0 - ARI(ground_truth, predicted_clusters)
            nmi_loss = 1.0 - NMI(ground_truth, predicted_clusters)

            # Combined loss
            loss = mse + kl + ae_loss + contrastive_loss + 0.1 * ari_loss + 0.1 * nmi_loss


        scaler.scale(loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update()

        loss_value = loss.item()
        loss_history.append(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(MODEL.state_dict(), f'{FOLDER}/models/best_model_{NAME}_{DATASET_ID}.pth')
            with open(f'{FOLDER}/models/best_model_{NAME}_{DATASET_ID}.pkl', 'wb') as f:
                pickle.dump(MODEL, f)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return loss_history
