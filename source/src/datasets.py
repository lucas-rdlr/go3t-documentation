import scipy
import numpy as np
import scanpy as sc

import torch
from torch_geometric.data import Data
# import pysodb

import jax.numpy as jnp
from wassersteinwormhole import Wormhole
from sklearn.neighbors import kneighbors_graph

from src.classes import DefaultConfig


def load_dataset(ID, VARIABLE=False):
    path = '/storage/data/spRNA-seq/SDMBench/'
    adata = sc.read_h5ad(path+ID+'.h5ad')

    print('Original shape of adata: ', adata.shape)

    # # Remove genes (columns) with NaN values
    # adata = adata[:, ~np.isnan(adata.X).any(axis=0)]
    # # Remove cells (rows) with NaN values
    # adata = adata[~np.isnan(adata.X).any(axis=1), :]
    # Remove cells that don't have a ground truth annotation
    adata = adata[~adata.obs['ground_truth'].isna(),:]

    print('Shape of adata after taking out NaN values: ', adata.shape)

    adata.var_names_make_unique()
    # sc.pp.filter_genes(adata, min_counts=10)
    # sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if VARIABLE:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']==True]

    # Assuming 'ground_truth' is the ground truth column in the .obs annotation
    ground_truth_labels = adata.obs['ground_truth'].astype('category').cat.codes.values
    # Encode the labels as integers
    adata.obs['labels'] = ground_truth_labels

    return adata


def load_dataset_maynard(ID, VARIABLE=False):
    sodb = pysodb.SODB()
    adata = sodb.load_experiment('maynard2021trans', ID)
    adata = adata[np.logical_not(adata.obs['Region'].isna())]  # Remove NaN

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if VARIABLE:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']==True]

    # Assuming 'Region' is the ground truth
    ground_truth_labels = adata.obs['Region'].astype('category').cat.codes.values
    adata.obs['Region'] = ground_truth_labels  # Encode the labels as integers

    return adata


def prepare_data_wassersteinwormhole(adata, pca, n_neighbors=15, device='cpu', alpha=0.5):
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    adata.obsm['X_pca'] = pca.fit_transform(adata.X)
    expression_knn = kneighbors_graph(adata.obsm['X_pca'], n_neighbors=n_neighbors, include_self=True, mode='connectivity')
    edge_index_expression = torch.tensor(np.array(expression_knn.nonzero()), dtype=torch.long).view(2, -1)
    edge_weight_expression = torch.ones(edge_index_expression.shape[1], device=device)

    spatial_coords = adata.obsm['spatial']
    features = adata.obsm['X_pca']

    # Prepare Wormhole data
    point_clouds = [jnp.array(spatial_coords)]  # Convert to JAX array
    weights = [jnp.ones(spatial_coords.shape[0]) / spatial_coords.shape[0]]  # Convert to JAX array

    # Initialize the Wormhole model
    wormhole_model = Wormhole(point_clouds, weights, config=DefaultConfig)

    # Use the wormhole model to compute distances
    spatial_coords_jnp = jnp.array(spatial_coords)
    features_jnp = jnp.array(features)

    # Ensure the dimensions are correct
    spatial_coords_jnp = jnp.expand_dims(spatial_coords_jnp, axis=0)
    features_jnp = jnp.expand_dims(features_jnp, axis=0)

    # Ensure weights have the correct shape
    weights_jnp = weights[0].flatten()

    # Debugging shapes
    print("Spatial coords shape:", spatial_coords_jnp.shape)
    print("Features shape:", features_jnp.shape)
    print("Weights shape:", weights_jnp.shape)

    # Ensure weights have the correct dimensions for vmap
    weights_jnp = jnp.expand_dims(weights_jnp, axis=0)

    spatial_distances = wormhole_model.jit_dist_enc((spatial_coords_jnp, weights_jnp), (spatial_coords_jnp, weights_jnp), DefaultConfig.eps_enc, DefaultConfig.lse_enc)
    feature_distances = wormhole_model.jit_dist_enc((features_jnp, weights_jnp), (features_jnp, weights_jnp), DefaultConfig.eps_enc, DefaultConfig.lse_enc)

    combined_distances = (1 - alpha) * spatial_distances + alpha * feature_distances
    threshold = np.median(combined_distances)
    significant_edges = (combined_distances < threshold)

    if significant_edges.any():
        edge_indices = torch.nonzero(significant_edges, as_tuple=False).t()
        edge_weight_spatial = torch.ones(edge_indices.shape[1], device=device)
    else:
        edge_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight_spatial = torch.empty((0,), dtype=torch.float, device=device)

    if edge_indices.shape[1] > 0:
        edge_index = torch.cat([edge_index_expression, edge_indices], dim=1)
        edge_weight = torch.cat([edge_weight_expression, edge_weight_spatial])
    else:
        edge_index = edge_index_expression
        edge_weight = edge_weight_expression

    data = Data(x=torch.tensor(features, dtype=torch.float32, device=device),
                edge_index=edge_index,
                edge_attr=edge_weight)

    return data