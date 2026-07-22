import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

class FeatureSimilarityCalculator:
    """
    Computes similarity/distance matrices for fixed-length engineered feature vectors.
    Replaces DTW/SBD because features are already fixed-length and time-invariant.
    """
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path (str): Path to the feature_motifs.csv dataset.
        """
        # 1. Load the features dataset
        self.df = pd.read_csv(csv_path)
        
        # 2. Separate metadata (IDs) from numerical features
        self.metadata = self.df[['Motif_ID', 'Signal_ID']]
        
        # Get all numeric column names (ignoring the text-based IDs)
        feature_cols = [col for col in self.df.columns if col not in ['Motif_ID', 'Signal_ID']]
        self.raw_features = self.df[feature_cols].values
        self.n_motifs = len(self.raw_features)
        
        # 3. CRITICAL: Standardize features (Mean=0, Std=1)
        # This prevents large values (like Shape_Energy) from dominating 
        # small values (like TS_ACF_Lag1) in the distance calculations.
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.raw_features)

    # ==========================================
    # STANDARD DISTANCE MEASURES
    # ==========================================
    def compute_euclidean(self) -> np.ndarray:
        """Standard straight-line distance in the multi-dimensional feature space."""
        return squareform(pdist(self.scaled_features, metric='euclidean'))

    def compute_manhattan(self) -> np.ndarray:
        """Cityblock (L1) distance, more robust to feature outliers."""
        return squareform(pdist(self.scaled_features, metric='cityblock'))

    def compute_cosine(self) -> np.ndarray:
        """
        Measures the angle between feature vectors. 
        Distance = 1 - Cosine Similarity.
        """
        return squareform(pdist(self.scaled_features, metric='cosine'))

    def compute_correlation(self) -> np.ndarray:
        """Pearson correlation distance between feature profiles."""
        return squareform(pdist(self.scaled_features, metric='correlation'))


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Initialize with features CSV
    # Note: Using file name 'future_motifs.csv'
    calculator = FeatureSimilarityCalculator("data/future_motifs.csv")
    
    print(f"Loaded {calculator.n_motifs} motifs for similarity computation.")
    print("\n--- Computing Feature Distance Matrices ---")
    
    # 2. Compute Distances
    euclidean_dist = calculator.compute_euclidean()
    print(f"Euclidean Matrix Shape: {euclidean_dist.shape}")
    
    cosine_dist = calculator.compute_cosine()
    print(f"Cosine Matrix Shape: {cosine_dist.shape}")
    
    correlation_dist = calculator.compute_correlation()
    print(f"Correlation Matrix Shape: {correlation_dist.shape}")
    
    # 3. Analyze output
    print("\n--- Example Distances ---")
    print(f"Comparing Motif 0 and Motif 1:")
    print(f"  Euclidean distance: {euclidean_dist[0, 1]:.4f}")
    print(f"  Cosine distance:    {cosine_dist[0, 1]:.4f}")