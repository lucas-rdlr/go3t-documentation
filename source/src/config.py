import json
import torch
import torch.optim as optim
from sklearn.decomposition import PCA

from src.classes import GNNMoE, GraphAutoencoder, NOTAE, SCARF

def general_params(path):

    with open(path, 'r') as file:
        params = json.load(file)
    
    datas_id = params['datas_id']
    n_clusters = params['n_clusters']
    batch_sizes = params['batch_sizes']
    folder = params['folder']
    date = params['date']
    variable_genes = params['variable_genes']

    if variable_genes:
        variable = 'Var_'

    else:
        variable = ''

    return datas_id, batch_sizes, folder, date, variable_genes, variable


def training_params(path):

    with open(path, 'r') as file:
        params = json.load(file)

    n_components = params['training']['pca']
    pca = PCA(n_components)
    epochs = params['training']['epochs']
    device = torch.device(
        f"cuda:{params['training']['device']}" if torch.cuda.is_available() else "cpu"
    )

    # device = torch.device(
    #     f"cuda:3" if torch.cuda.is_available() else "cpu"
    # )

    hidden_model = params['training']['model']['hidden']
    heads = params['training']['model']['heads']
    num_experts = params['training']['model']['num_experts']
    model = GNNMoE(n_components, hidden_model, n_components, heads=heads, num_experts=num_experts).to(device)
    
    hidden_auto = params['training']['autoencoder']['hidden']
    autoencoder = NOTAE(n_components, hidden_auto, n_components).to(device)

    lr = params['training']['optimizer']['lr']
    weight_decay = params['training']['optimizer']['weight_decay']
    optimizer = optim.Adam(list(model.parameters()) + list(autoencoder.parameters()), lr=lr, weight_decay=weight_decay)

    return pca, epochs, device, model, autoencoder, optimizer