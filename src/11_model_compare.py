import pandas as pd
import numpy as np

class MotifModelComparator:
    """
    Aggregates evaluation metrics from all tested clustering models, 
    generates a standardized comparison table, and automatically selects 
    the best-performing model based on the specified internal validation metric.
    """
    def __init__(self):
        self.results = []

    def add_model_result(self, model_name: str, silhouette: float, 
                         db_index: float, ch_score: float, 
                         runtime: float, num_clusters: int):
        """
        Appends the evaluation metrics of a single model to the comparator.
        """
        self.results.append({
            'Model': model_name,
            'Silhouette': silhouette,
            'DB Index': db_index,
            'CH Score': ch_score,
            'Runtime (s)': runtime,
            'Number of Clusters': num_clusters
        })

    def get_comparison_table(self) -> pd.DataFrame:
        """Returns the aggregated results as a formatted Pandas DataFrame."""
        if not self.results:
            raise ValueError("No model results have been added yet.")
            
        df = pd.DataFrame(self.results)
        
        # Clean formatting for display purposes
        df['Silhouette'] = df['Silhouette'].round(4)
        df['DB Index'] = df['DB Index'].round(4)
        df['CH Score'] = df['CH Score'].round(2)
        df['Runtime (s)'] = df['Runtime (s)'].round(4)
        
        return df

    def select_best_model(self, primary_metric: str = 'Silhouette') -> pd.Series:
        """
        Automatically evaluates the dataframe to find the best model.
        - Silhouette: Higher is better (Range: -1 to 1)
        - DB Index: Lower is better (Range: 0 to infinity)
        - CH Score: Higher is better (Range: 0 to infinity)
        """
        df = self.get_comparison_table()
        
        # Filter out models that failed to cluster properly (e.g., all noise or single cluster)
        # In our pipeline, failed silhouette scores are designated as -1.0
        valid_df = df[df['Silhouette'] > -1.0]
        
        if valid_df.empty:
            raise ValueError("No valid models found. All algorithms failed to form distinct clusters.")
            
        if primary_metric == 'Silhouette':
            best_idx = valid_df['Silhouette'].idxmax()
        elif primary_metric == 'DB Index':
            best_idx = valid_df['DB Index'].idxmin()
        elif primary_metric == 'CH Score':
            best_idx = valid_df['CH Score'].idxmax()
        else:
            raise ValueError("Unsupported primary metric. Choose 'Silhouette', 'DB Index', or 'CH Score'.")
            
        best_model = valid_df.loc[best_idx]
        return best_model

# ==========================================
# EXECUTION & SAMPLE OUTPUT
# ==========================================
if __name__ == "__main__":
    comparator = MotifModelComparator()
    
    # Simulating the data collection from your pipeline runs.
    # In production, these metrics would be pulled directly from the outputs of Stage 6 and Stage 8.
    
    # 1. Centroid Models
    comparator.add_model_result("KMeans", 0.6214, 0.4512, 1420.5, 0.1502, 5)
    
    # 2. Hierarchical Models
    comparator.add_model_result("Agglomerative", 0.5982, 0.5104, 1310.2, 0.8541, 5)
    
    # 3. Density-Based Models
    # DBSCAN often struggles with varying densities, resulting in suboptimal scores or many noise points
    comparator.add_model_result("DBSCAN", 0.3120, 1.8401, 210.4, 0.4120, 12) 
    comparator.add_model_result("HDBSCAN", 0.5421, 0.7612, 980.6, 1.2501, 6)
    
    # 4. Graph Models
    comparator.add_model_result("Spectral", 0.6055, 0.4921, 1390.8, 1.8903, 5)
    
    # 5. SOM
    comparator.add_model_result("SOM", 0.5112, 0.8201, 850.3, 2.4501, 9)
    
    # 6. Deep Learning Models
    # AE+KMeans relies on latent space, DEC optimizes the latent space specifically for clustering
    comparator.add_model_result("Autoencoder + KMeans", 0.6540, 0.4102, 1550.6, 45.2010, 5)
    comparator.add_model_result("DEC (Deep Embedded)", 0.6892, 0.3510, 1720.9, 125.6020, 5)
    
    # Retrieve and print the formatted table
    comparison_df = comparator.get_comparison_table()
    print("=== GLOBAL MODEL COMPARISON TABLE ===\n")
    print(comparison_df.to_markdown(index=False))
    
    # Auto-Select the best model
    print("\n\n=== AUTOMATED MODEL SELECTION ===")
    best_model = comparator.select_best_model(primary_metric='Silhouette')
    
    print(f"[*] Evaluation Criteria: Maximize Silhouette Score")
    print(f"[*] Best Model Selected: {best_model['Model']}")
    print(f"    -> Silhouette Score: {best_model['Silhouette']}")
    print(f"    -> Davies-Bouldin:   {best_model['DB Index']}")
    print(f"    -> Compute Time:     {best_model['Runtime (s)']} seconds")
    print(f"    -> Final Clusters:   {best_model['Number of Clusters']}")