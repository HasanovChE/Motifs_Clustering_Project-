import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# Scikit-Learn Clustering
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, 
    OPTICS, SpectralClustering, Birch, AffinityPropagation, MeanShift
)
from sklearn.mixture import GaussianMixture

# Requires scikit-learn >= 1.3 for built-in HDBSCAN. 
# Alternatively: pip install hdbscan
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    import hdbscan
    HDBSCAN = hdbscan.HDBSCAN

# # Requires: pip install scikit-learn-extra
# from sklearn_extra.cluster import KMedoids

# Requires: pip install minisom
from minisom import MiniSom

warnings.filterwarnings("ignore")

# ==========================================
# DEEP LEARNING ARCHITECTURES (PyTorch)
# ==========================================
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class ClusteringLayer(nn.Module):
    """
    Computes the soft assignment (Student's t-distribution) of latent representations 
    to cluster centers. Core component for Deep Embedded Clustering (DEC).
    """
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


# ==========================================
# CLUSTERING PIPELINE
# ==========================================
class GlobalMotifClusterer:
    """
    A comprehensive suite of clustering algorithms for time-series motifs.
    Accepts either feature matrices (n_samples, n_features) or 
    precomputed distance matrices (n_samples, n_samples).
    """
    def __init__(self, data: np.ndarray, is_precomputed: bool = False):
        self.data = data
        self.is_precomputed = is_precomputed
        self.n_samples = data.shape[0]

    def _check_precomputed(self, algo_name: str, requires_features: bool = False):
        if self.is_precomputed and requires_features:
            raise ValueError(f"{algo_name} requires a feature matrix, but a distance matrix was provided.")

    # ------------------------------------------
    # 1. CENTROID-BASED MODELS
    # ------------------------------------------
    def run_kmeans(self, n_clusters=5, init='k-means++', max_iter=300):
        self._check_precomputed("KMeans", requires_features=True)
        model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
        return model.fit_predict(self.data)

    def run_minibatch_kmeans(self, n_clusters=5, batch_size=1024):
        self._check_precomputed("MiniBatch KMeans", requires_features=True)
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
        return model.fit_predict(self.data)

    # def run_kmedoids(self, n_clusters=5, method='pam'):
    #     """K-Medoids is excellent for precomputed DTW matrices."""
    #     metric = 'precomputed' if self.is_precomputed else 'euclidean'
    #     model = KMedoids(n_clusters=n_clusters, metric=metric, method=method, random_state=42)
    #     return model.fit_predict(self.data)

    # ------------------------------------------
    # 2. HIERARCHICAL MODELS
    # ------------------------------------------
    def run_agglomerative(self, n_clusters=5, linkage='ward'):
        """Linkages: 'ward', 'average', 'complete', 'single'"""
        if self.is_precomputed and linkage == 'ward':
            print("Warning: Ward linkage requires euclidean distance. Switching to 'average'.")
            linkage = 'average'
            
        metric = 'precomputed' if self.is_precomputed else 'euclidean'
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        return model.fit_predict(self.data)

    def run_birch(self, n_clusters=5, threshold=0.5):
        self._check_precomputed("Birch", requires_features=True)
        model = Birch(n_clusters=n_clusters, threshold=threshold)
        return model.fit_predict(self.data)

    # ------------------------------------------
    # 3. DENSITY-BASED MODELS
    # ------------------------------------------
    def run_dbscan(self, eps=0.5, min_samples=5):
        metric = 'precomputed' if self.is_precomputed else 'euclidean'
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        return model.fit_predict(self.data)

    def run_hdbscan(self, min_cluster_size=5, min_samples=None):
        metric = 'precomputed' if self.is_precomputed else 'euclidean'
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        return model.fit_predict(self.data)

    def run_optics(self, min_samples=5, xi=0.05):
        metric = 'precomputed' if self.is_precomputed else 'euclidean'
        model = OPTICS(min_samples=min_samples, xi=xi, metric=metric)
        return model.fit_predict(self.data)

    def run_meanshift(self, bandwidth=None):
        self._check_precomputed("MeanShift", requires_features=True)
        model = MeanShift(bandwidth=bandwidth)
        return model.fit_predict(self.data)

    # ------------------------------------------
    # 4. GRAPH & PROBABILISTIC MODELS
    # ------------------------------------------
    def run_spectral(self, n_clusters=5):
        affinity = 'precomputed' if self.is_precomputed else 'rbf'
        if self.is_precomputed:
            # Spectral requires an affinity/similarity matrix (higher is better), not a distance matrix.
            # Convert distance to similarity using a Gaussian kernel.
            gamma = 1.0 / self.data.std()
            sim_matrix = np.exp(-gamma * (self.data ** 2))
            model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
            return model.fit_predict(sim_matrix)
            
        model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
        return model.fit_predict(self.data)

    def run_gmm(self, n_components=5):
        self._check_precomputed("GMM", requires_features=True)
        model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        return model.fit_predict(self.data)

    def run_affinity_propagation(self):
        affinity = 'precomputed' if self.is_precomputed else 'euclidean'
        if self.is_precomputed:
             # AP expects similarities (negative distances)
            sim_matrix = -self.data
            model = AffinityPropagation(affinity='precomputed', random_state=42)
            return model.fit_predict(sim_matrix)
            
        model = AffinityPropagation(random_state=42)
        return model.fit_predict(self.data)

    # ------------------------------------------
    # 5. SELF-ORGANIZING MAPS (SOM)
    # ------------------------------------------
    def run_som(self, grid_x=10, grid_y=10, sigma=1.0, learning_rate=0.5, epochs=1000):
        self._check_precomputed("SOM", requires_features=True)
        # Standardize data for SOM
        data_norm = (self.data - np.mean(self.data, axis=0)) / (np.std(self.data, axis=0) + 1e-8)
        n_features = data_norm.shape[1]
        
        som = MiniSom(grid_x, grid_y, n_features, sigma=sigma, learning_rate=learning_rate)
        som.random_weights_init(data_norm)
        som.train_random(data_norm, epochs)
        
        # Map each sample to its winning neuron (x, y) coordinates
        cluster_labels = []
        for x in data_norm:
            winner = som.winner(x)
            cluster_labels.append(winner[0] * grid_y + winner[1]) # Flatten grid to 1D label
        return np.array(cluster_labels)

    # ------------------------------------------
    # 6. DEEP LEARNING (AE + KMeans / DEC)
    # ------------------------------------------
    def run_autoencoder_kmeans(self, n_clusters=5, latent_dim=10, epochs=50, batch_size=64):
        self._check_precomputed("Autoencoder", requires_features=True)
        input_dim = self.data.shape[1]
        model = SimpleAutoencoder(input_dim, latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        tensor_data = torch.FloatTensor(self.data)
        loader = DataLoader(TensorDataset(tensor_data, tensor_data), batch_size=batch_size, shuffle=True)
        
        # Train AE
        model.train()
        for epoch in range(epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstruction = model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                
        # Extract latent features
        model.eval()
        with torch.no_grad():
            latent_features = model.encoder(tensor_data).numpy()
            
        # Run standard KMeans on latent space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(latent_features)

    def run_dec(self, n_clusters=5, latent_dim=10, pretrain_epochs=30, cluster_epochs=50):
        """Deep Embedded Clustering (DEC)"""
        self._check_precomputed("DEC", requires_features=True)
        input_dim = self.data.shape[1]
        ae = SimpleAutoencoder(input_dim, latent_dim)
        tensor_data = torch.FloatTensor(self.data)
        
        # 1. Pretrain Autoencoder
        optimizer_ae = optim.Adam(ae.parameters(), lr=1e-3)
        criterion_mse = nn.MSELoss()
        for _ in range(pretrain_epochs):
            optimizer_ae.zero_grad()
            loss = criterion_mse(ae(tensor_data), tensor_data)
            loss.backward()
            optimizer_ae.step()
            
        # 2. Initialize Cluster Centers via KMeans
        ae.eval()
        with torch.no_grad():
            latent_features = ae.encoder(tensor_data)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        kmeans.fit(latent_features.numpy())
        
        # 3. Setup DEC Clustering Layer
        cluster_layer = ClusteringLayer(n_clusters, latent_dim)
        cluster_layer.centers.data = torch.tensor(kmeans.cluster_centers_)
        
        # 4. Joint Training (Minimize KL Divergence)
        def target_distribution(q):
            weight = (q ** 2) / torch.sum(q, dim=0)
            return (weight.t() / torch.sum(weight, dim=1)).t()

        optimizer_dec = optim.Adam(list(ae.encoder.parameters()) + list(cluster_layer.parameters()), lr=1e-4)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        ae.train()
        for _ in range(cluster_epochs):
            optimizer_dec.zero_grad()
            z = ae.encoder(tensor_data)
            q = cluster_layer(z)
            p = target_distribution(q).detach()
            
            loss = criterion_kl(torch.log(q + 1e-8), p)
            loss.backward()
            optimizer_dec.step()
            
        # Final prediction
        with torch.no_grad():
            q = cluster_layer(ae.encoder(tensor_data))
            preds = torch.argmax(q, dim=1).numpy()
            
        return preds


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure dependencies: 
    # pip install scikit-learn scikit-learn-extra hdbscan minisom torch
    
    print("Loading feature matrix and computing distance matrix...")

    # 1. Load your actual dataset
    df = pd.read_csv("data/future_motifs.csv")

    # 2. Extract only the numerical features (ignore the string IDs)
    feature_cols = [col for col in df.columns if col not in ['Motif_ID', 'Signal_ID']]
    raw_features = df[feature_cols].values

    # 3. Create the real `feature_matrix` (Standardized)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(raw_features)

    # 4. Create the real `dist_matrix` (using Euclidean distance)
    distances = pdist(feature_matrix, metric='euclidean')
    dist_matrix = squareform(distances)

    print(f"Feature Matrix Shape: {feature_matrix.shape}")
    print(f"Distance Matrix Shape: {dist_matrix.shape}")
    
    # 1. Feature-based clustering (e.g., Output from Stage 4)
    print("\n--- Running Feature-Based Algorithms ---")
    clusterer_features = GlobalMotifClusterer(feature_matrix, is_precomputed=False)
    
    kmeans_labels = clusterer_features.run_kmeans(n_clusters=5)
    print(f"KMeans unique clusters: {np.unique(kmeans_labels)}")
    
    ae_labels = clusterer_features.run_autoencoder_kmeans(n_clusters=5, epochs=5)
    print(f"Autoencoder+KMeans unique clusters: {np.unique(ae_labels)}")
    
    dec_labels = clusterer_features.run_dec(n_clusters=5, pretrain_epochs=5, cluster_epochs=5)
    print(f"DEC unique clusters: {np.unique(dec_labels)}")

    # 2. Distance-based clustering (e.g., Output from Stage 5 - DTW/Hybrid)
    print("\n--- Running Precomputed Distance Algorithms ---")
    clusterer_dist = GlobalMotifClusterer(dist_matrix, is_precomputed=True)
    
    # kmedoids_labels = clusterer_dist.run_kmedoids(n_clusters=5)
    # print(f"K-Medoids unique clusters: {np.unique(kmedoids_labels)}")
    
    hdbscan_labels = clusterer_dist.run_hdbscan(min_cluster_size=10)
    print(f"HDBSCAN unique clusters (includes -1 for noise): {np.unique(hdbscan_labels)}")