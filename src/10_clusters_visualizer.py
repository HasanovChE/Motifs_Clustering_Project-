import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import umap

class MotifVisualizer:
    """
    Comprehensive visualization suite for time-series motif clustering.
    Handles signal overlays, dimensionality reduction, and cluster diagnostics.
    """
    def __init__(self, style='whitegrid'):
        sns.set_theme(style=style)
        self.palette = sns.color_palette("tab10", 10)

    # ==========================================
    # 1. SIGNAL & MOTIF VISUALIZATIONS
    # ==========================================
    def plot_vertical_signal_with_clusters(self, signal: np.ndarray, 
                                           motif_starts: list, motif_ends: list, 
                                           labels: list, signal_name: str = "Signal"):
        """
        Plots the raw signal vertically (depth-log style) and shades the detected 
        motif regions using colors corresponding to their assigned cluster.
        Matches the style of Figure 1.
        """
        fig, ax = plt.subplots(figsize=(4, 12))
        idx = np.arange(len(signal))
        
        # Plot raw signal
        ax.plot(signal, idx, color='steelblue', linewidth=0.8)
        
        # Overlay motifs with cluster-specific colors
        unique_labels = set(labels)
        color_map = {lbl: self.palette[lbl % len(self.palette)] if lbl != -1 else 'gray' 
                     for lbl in unique_labels}
        
        for start, end, label in zip(motif_starts, motif_ends, labels):
            color = color_map[label]
            ax.axhspan(start, end, color=color, alpha=0.3)
            
            # Highlight the motif line itself
            ax.plot(signal[start:end], idx[start:end], color=color, linewidth=1.5)

        ax.invert_yaxis() # Index 0 at the top
        ax.set_title(f"{signal_name} w/ Clustered Motifs")
        ax.set_ylabel("Index / Depth")
        ax.set_xlabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def plot_generalized_motifs(self, motifs_list: list, labels: list, target_length: int = 100):
        """
        Recreates Figure 2. Interpolates all motifs in a cluster to a uniform length,
        plots them in light grey, and overlays the mean (generalized) motif in bold color.
        """
        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1] # Ignore noise
        n_clusters = len(unique_labels)
        
        cols = min(3, n_clusters)
        rows = int(np.ceil(n_clusters / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        axes = axes.flatten()
        
        for i, cluster_id in enumerate(unique_labels):
            ax = axes[i]
            cluster_motifs = [motifs_list[j] for j in range(len(motifs_list)) if labels[j] == cluster_id]
            
            resampled = []
            x_new = np.linspace(0, 1, target_length)
            
            # Plot individual motifs in background
            for motif in cluster_motifs:
                x_old = np.linspace(0, 1, len(motif))
                interpolator = interp1d(x_old, motif, kind='linear')
                resampled_motif = interpolator(x_new)
                resampled.append(resampled_motif)
                
                # Plot horizontal by default for generalized view, or swap (x_new, resampled_motif) for vertical
                ax.plot(x_new, resampled_motif, color='gray', alpha=0.15, linewidth=1)
                
            # Plot generalized (mean) motif
            if resampled:
                mean_motif = np.mean(resampled, axis=0)
                color = self.palette[cluster_id % len(self.palette)]
                ax.plot(x_new, mean_motif, color=color, linewidth=3, label=f'Cluster {cluster_id}')
            
            ax.set_title(f"Cluster {cluster_id} (n={len(cluster_motifs)})")
            ax.set_xticks([]) # Remove x-ticks to focus on shape
            
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 2. HIERARCHICAL & DISTANCE VISUALIZATIONS
    # ==========================================
    def plot_dendrogram(self, X: np.ndarray, truncate_mode='level', p=5):
        """Plots the hierarchical clustering dendrogram."""
        fig, ax = plt.subplots(figsize=(10, 5))
        Z = linkage(X, method='ward')
        dendrogram(Z, truncate_mode=truncate_mode, p=p, ax=ax, 
                   leaf_rotation=90., leaf_font_size=10., show_contracted=True)
        ax.set_title("Hierarchical Clustering Dendrogram (Truncated)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.show()

    def plot_distance_matrix_heatmap(self, dist_matrix: np.ndarray, labels: list):
        """Plots the distance matrix, sorted by cluster assignments to show block structures."""
        # Sort indices by cluster label
        sorted_indices = np.argsort(labels)
        sorted_dist = dist_matrix[sorted_indices, :][:, sorted_indices]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(sorted_dist, cmap='viridis_r', xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title("Distance Matrix Heatmap (Sorted by Cluster)")
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 3. DIMENSIONALITY REDUCTION (PCA, t-SNE, UMAP)
    # ==========================================
    def plot_dimensionality_reduction(self, X: np.ndarray, labels: list, method='tsne'):
        """Projects high-dimensional feature vectors into 2D using PCA, t-SNE, or UMAP."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            title = "PCA 2D Projection"
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            title = "t-SNE 2D Projection"
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            title = "UMAP 2D Projection"
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
            
        embedding = reducer.fit_transform(X)
        
        # Scatter plot colored by labels
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, 
                             cmap='tab10', alpha=0.7, edgecolors='w', s=50)
        
        # Add legend
        unique_lbls = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(l)), 
                              markersize=8, label=f'Cluster {l}' if l != -1 else 'Noise') for l in unique_lbls]
        ax.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 4. DIAGNOSTIC PLOTS
    # ==========================================
    def plot_cluster_length_histograms(self, lengths: list, labels: list):
        """Plots the distribution of motif lengths within each cluster."""
        df = pd.DataFrame({'Length': lengths, 'Cluster': labels})
        df = df[df['Cluster'] != -1] # Exclude noise
        
        g = sns.FacetGrid(df, col="Cluster", col_wrap=3, height=4, sharex=False, sharey=False)
        g.map_dataframe(sns.histplot, x="Length", kde=True, bins=15, color='teal')
        g.set_axis_labels("Motif Length", "Count")
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle("Motif Length Distributions per Cluster")
        plt.show()

    def plot_silhouette(self, X: np.ndarray, labels: list):
        """Generates a silhouette plot for cluster validation."""
        valid_idx = np.array(labels) != -1
        if not np.any(valid_idx) or len(np.unique(np.array(labels)[valid_idx])) < 2:
            print("Not enough valid clusters for Silhouette Plot.")
            return

        clean_X = X[valid_idx]
        clean_labels = np.array(labels)[valid_idx]
        
        n_clusters = len(np.unique(clean_labels))
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(clean_X) + (n_clusters + 1) * 10])

        silhouette_avg = silhouette_score(clean_X, clean_labels)
        sample_values = silhouette_samples(clean_X, clean_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_values = sample_values[clean_labels == i]
            ith_cluster_values.sort()

            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = self.palette[i % len(self.palette)]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # Space between clusters

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_title("Silhouette Plot for Various Clusters")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        
        # Ticks optimization
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.1, 1.1, 0.2))
        plt.tight_layout()
        plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Dependencies: pip install matplotlib seaborn scipy scikit-learn umap-learn

    print("Loading real data for visualizations...")

    df_features = pd.read_csv("data/future_motifs.csv")

    # Extract and scale numeric features
    feature_cols = [col for col in df_features.columns if col not in ['Motif_ID', 'Signal_ID']]
    X_features = StandardScaler().fit_transform(df_features[feature_cols].values)

    # Compute real distance matrix
    dist_mat = pairwise_distances(X_features, metric='euclidean')

    # Run K-Means to get labels for coloring the plots
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    motif_labels = kmeans.fit_predict(X_features)
    motif_lengths = df_features['Shape_Length'].values.astype(int)

    df_motifs = pd.read_csv("data/cleaned_motifs.csv")

    # Safely parse the text strings back into numpy arrays
    df_motifs['Raw_Signal'] = df_motifs['Raw_Signal'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    motifs = [np.array(m) for m in df_motifs['Raw_Signal']]

    try:
      df_continuous = pd.read_csv("data/preprocessed_signals_data.csv")
      # Grab the very first signal in the dataset to act as our visualization background
      first_sig_col = df_continuous.columns[0]
      raw_signal = df_continuous[first_sig_col].values
    
      # Filter the starts/ends/labels to ONLY include motifs from this specific signal
      # This prevents the timeline plot from crashing by trying to plot motifs from other signals
      sig_mask = df_motifs['Signal_ID'] == first_sig_col
    
      starts = df_motifs.loc[sig_mask, 'Start_Index'].values.astype(int)
      ends = df_motifs.loc[sig_mask, 'End_Index'].values.astype(int)
      signal_labels = motif_labels[sig_mask]
    
    except FileNotFoundError:
      print("WARNING: 'data/cleaned_signals_data.csv' not found. Using a fallback signal for the timeline.")
      # Fallback just in case the continuous data isn't saved in this directory
      raw_signal = np.zeros(1000)
      starts, ends, signal_labels = [], [], []

      print(f"Visualization data ready: {len(motifs)} motifs loaded across {len(np.unique(motif_labels))} clusters.")
    
    # Initialize Visualizer
    viz = MotifVisualizer()
    
    print("\n--- Plotting Vertical Signal Track ---")
    viz.plot_vertical_signal_with_clusters(raw_signal, starts, ends, signal_labels)
    
    print("\n--- Plotting Generalized Motifs (Fig 2 style) ---")
    viz.plot_generalized_motifs(motifs, motif_labels)
    
    print("\n--- Plotting Dendrogram ---")
    viz.plot_dendrogram(X_features)
    
    print("\n--- Plotting Distance Matrix Heatmap ---")
    viz.plot_distance_matrix_heatmap(dist_mat, sorted(motif_labels))
    
    print("\n--- Plotting Dimensionality Reductions ---")
    viz.plot_dimensionality_reduction(X_features, sorted(motif_labels), method='pca')
    viz.plot_dimensionality_reduction(X_features, sorted(motif_labels), method='tsne')
    viz.plot_dimensionality_reduction(X_features, sorted(motif_labels), method='umap')
        
    print("\n--- Plotting Diagnostics ---")
    viz.plot_cluster_length_histograms(motif_lengths, motif_labels)
    viz.plot_silhouette(X_features, sorted(motif_labels))