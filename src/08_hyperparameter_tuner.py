import numpy as np
import pandas as pd
import itertools
from typing import Dict, Any, Callable, Tuple
import warnings
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import silhouette_score
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
)
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except (ImportError, ModuleNotFoundError, Exception):
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
        HAS_HDBSCAN = True
    except (ImportError, ModuleNotFoundError, Exception):
        HDBSCAN = None
        HAS_HDBSCAN = False

warnings.filterwarnings("ignore")

class ClusteringOptimizer:
    """
    Hyperparameter tuning framework for unsupervised clustering algorithms.
    Optimizes parameters based on maximizing the Silhouette Score.
    """
    def __init__(self, data: np.ndarray, is_precomputed: bool = False):
        self.data = data
        self.is_precomputed = is_precomputed
        self.metric = 'precomputed' if is_precomputed else 'euclidean'

    def _evaluate_labels(self, labels: np.ndarray) -> float:
        """
        Calculates the silhouette score. 
        Returns -1.0 (worst score) if valid clusters cannot be formed (e.g., all noise).
        """
        # Filter out noise (-1) for evaluation in density-based algorithms
        unique_labels = np.unique(labels)
        
        # Silhouette requires at least 2 distinct clusters (excluding noise)
        valid_labels = unique_labels[unique_labels != -1]
        if len(valid_labels) < 2:
            return -1.0
            
        try:
            return silhouette_score(self.data, labels, metric=self.metric)
        except ValueError:
            return -1.0

    def grid_search(self, model_class: Callable, param_grid: Dict[str, list], 
                    fixed_params: Dict[str, Any] = None) -> Tuple[Dict, float, np.ndarray]:
        """
        Executes an exhaustive grid search over the provided hyperparameter space.
        """
        fixed_params = fixed_params or {}
        keys, values = zip(*param_grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_score = -1.0
        best_params = None
        best_labels = None
        
        print(f"Running Grid Search over {len(permutations)} combinations...")
        
        for params in permutations:
            combined_params = {**params, **fixed_params}
            
            try:
                # Spectral Clustering needs similarity, not distance, if precomputed
                if model_class == SpectralClustering and self.is_precomputed:
                    gamma = combined_params.get('gamma', 1.0)
                    sim_matrix = np.exp(-gamma * (self.data ** 2))
                    model = model_class(**combined_params)
                    labels = model.fit_predict(sim_matrix)
                else:
                    model = model_class(**combined_params)
                    labels = model.fit_predict(self.data)
                
                score = self._evaluate_labels(labels)
                
                if score > best_score:
                    best_score = score
                    best_params = combined_params
                    best_labels = labels
            except Exception as e:
                # Catch invalid parameter combinations (e.g., HDBSCAN min_samples > min_cluster_size)
                continue
                
        return best_params, best_score, best_labels

    # ==========================================
    # ALGORITHM-SPECIFIC OPTIMIZERS
    # ==========================================
    def optimize_kmeans(self) -> Tuple[Dict, float]:
        if self.is_precomputed:
            print("Skipping KMeans (requires feature matrix, not precomputed distances).")
            return None, -1.0
            
        param_grid = {'n_clusters': list(range(2, 21))} # K = 2...20
        fixed = {'init': 'k-means++', 'max_iter': 300, 'random_state': 42}
        
        print("\n--- Optimizing KMeans ---")
        best_p, best_s, _ = self.grid_search(KMeans, param_grid, fixed)
        print(f"Best Params: {best_p} | Best Silhouette: {best_s:.4f}")
        return best_p, best_s

    def optimize_dbscan(self) -> Tuple[Dict, float]:
        param_grid = {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
            'min_samples': [2, 3, 5, 10]
        }
        fixed = {'metric': self.metric}
        
        print("\n--- Optimizing DBSCAN ---")
        best_p, best_s, _ = self.grid_search(DBSCAN, param_grid, fixed)
        print(f"Best Params: {best_p} | Best Silhouette: {best_s:.4f}")
        return best_p, best_s

    def optimize_agglomerative(self) -> Tuple[Dict, float]:
        param_grid = {
            # Use None for n_clusters if using distance_threshold
            'n_clusters': [None], 
            'distance_threshold': [0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
            'linkage': ['average', 'complete', 'single'] # Ward requires Euclidean
        }
        if not self.is_precomputed:
            param_grid['linkage'].append('ward')
            
        fixed = {'metric': self.metric}
        
        print("\n--- Optimizing Agglomerative ---")
        best_p, best_s, _ = self.grid_search(AgglomerativeClustering, param_grid, fixed)
        print(f"Best Params: {best_p} | Best Silhouette: {best_s:.4f}")
        return best_p, best_s

    def optimize_hdbscan(self) -> Tuple[Dict, float]:
        if not HAS_HDBSCAN or HDBSCAN is None:
            print("HDBSCAN not installed. Skipping HDBSCAN optimization.")
            return None, -1.0
        param_grid = {
            'min_cluster_size': [3, 5, 10, 15],
            'min_samples': [1, 3, 5, 10]
        }
        fixed = {'metric': self.metric}
        
        print("\n--- Optimizing HDBSCAN ---")
        best_p, best_s, _ = self.grid_search(HDBSCAN, param_grid, fixed)
        print(f"Best Params: {best_p} | Best Silhouette: {best_s:.4f}")
        return best_p, best_s

    def optimize_spectral(self) -> Tuple[Dict, float]:
        param_grid = {
            'n_clusters': list(range(2, 10)),
            'gamma': [0.1, 1.0, 10.0]
        }
        fixed = {
            'affinity': 'precomputed' if self.is_precomputed else 'rbf',
            'random_state': 42
        }
        
        print("\n--- Optimizing Spectral Clustering ---")
        best_p, best_s, _ = self.grid_search(SpectralClustering, param_grid, fixed)
        print(f"Best Params: {best_p} | Best Silhouette: {best_s:.4f}")
        return best_p, best_s

    def optimize_deep_learning_dec(self, ae_class, dec_runner) -> Tuple[Dict, float]:
        """
        Deep Learning Optimization requires a wrapper around the DEC function from Stage 6.
        We pass the pre-defined DEC runner function to evaluate different hyperparameter sets.
        """
        if self.is_precomputed:
            print("Skipping DEC (requires feature matrix).")
            return None, -1.0

        param_grid = {
            'latent_dim': [5, 10, 20],
            'learning_rate': [1e-3, 1e-4], # Typically handled inside the DEC runner wrapper
            'epochs': [30, 50],
            'batch_size': [32, 64]
        }
        
        keys, values = zip(*param_grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_score = -1.0
        best_params = None
        
        print(f"\n--- Optimizing Deep Embedded Clustering (DEC) ---")
        print(f"Running custom Grid Search over {len(permutations)} DL configurations...")
        
        for params in permutations:
            # Here, dec_runner is expected to accept these kwargs and return cluster labels
            try:
                labels = dec_runner(
                    self.data, 
                    latent_dim=params['latent_dim'], 
                    lr=params['learning_rate'], 
                    epochs=params['epochs'], 
                    batch_size=params['batch_size']
                )
                score = self._evaluate_labels(labels)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue
                
        print(f"Best Params: {best_params} | Best Silhouette: {best_score:.4f}")
        return best_params, best_score

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    df = pd.read_csv("data/future_motifs.csv")
    feature_cols = [col for col in df.columns if col not in ['Motif_ID', 'Signal_ID']]
    raw_features = df[feature_cols].values
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(raw_features)
    
    # 1. Optimize Feature-Based Algorithms
    optimizer = ClusteringOptimizer(feature_matrix, is_precomputed=False)
    
    optimizer.optimize_kmeans()
    optimizer.optimize_dbscan()
    optimizer.optimize_agglomerative()
    optimizer.optimize_spectral()