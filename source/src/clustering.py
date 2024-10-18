import numpy as np
import ot
from sklearn.decomposition import PCA
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


# based on GraphST
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='GO3T_pca', random_seed=2024):

    np.random.seed(random_seed)
    
    robjects.r.library('mclust')
    
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata 


def clustering(adata, n_clusters = 15, radius =50, key = 'GO3T', method = "mclust", start = 0.001, increment = 0.01, refinement=True):

    pca = PCA(n_components=20, random_state=2024)
    GO3T = pca.fit_transform(adata.obsm['GO3T'])
    adata.obsm['GO3T_pca'] = GO3T
    
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='GO3T_pca', num_cluster=n_clusters)
        adata.obs['prediction'] = adata.obs['mclust']
    
    if refinement:
        new_type = refine_label(adata, radius, key='prediction')
        adata.obs['prediction'] = new_type


def refine_label(adata, radius = 50, key='prediction'):

    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    
    new_type = [str(i) for i in list(new_type)]

    return new_type


def fine_tune_clustering_label_propagation(adata, key='mclust', n_neighbors=15, max_iter=300, tol=1e-3):
    # Initial clustering from mclust
    initial_labels = adata.obs[key].values
    refined_labels = initial_labels.copy()


    # Construct kNN graph
    knn_graph = kneighbors_graph(adata.obsm['spatial'], n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    G = nx.from_scipy_sparse_array(knn_graph)


    for iteration in range(max_iter):
        prev_labels = refined_labels.copy()
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_labels = refined_labels[neighbors]
                # Get the most frequent label among the neighbors
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                most_frequent_label = unique_labels[np.argmax(counts)]
                refined_labels[node] = most_frequent_label


        # Check for convergence
        if np.sum(prev_labels != refined_labels) / len(refined_labels) < tol:
            break


    # Update the adata object with new labels
    adata.obs['fine_tuned'] = refined_labels
    adata.obs['fine_tuned'] = adata.obs['fine_tuned'].astype('int')
    adata.obs['fine_tuned'] = adata.obs['fine_tuned'].astype('category')


    return adata


def optimize_n_neighbors(adata, key='mclust', n_neighbors_range=[5, 10, 15, 20, 25, 30], max_iter=300, tol=1e-3):
    best_n_neighbors = n_neighbors_range[0]
    best_ari = 0
    aris = []


    for n_neighbors in n_neighbors_range:
        print(f"Evaluating n_neighbors = {n_neighbors}")
        adata_temp = fine_tune_clustering_label_propagation(adata.copy(), key=key, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol)
        ari = adjusted_rand_score(adata.obs['Region'], adata_temp.obs['fine_tuned'])
        aris.append(ari)
        if ari > best_ari:
            best_ari = ari
            best_n_neighbors = n_neighbors


    #plt.figure(figsize=(10, 5))
    #plt.plot(n_neighbors_range, aris, marker='o')
    #plt.xlabel('Number of Neighbors')
    #plt.ylabel('ARI')
    #plt.title('Optimization of n_neighbors')
    #plt.show()


    print(f"Best n_neighbors: {best_n_neighbors}, Best ARI: {best_ari}")
    
    return best_n_neighbors
