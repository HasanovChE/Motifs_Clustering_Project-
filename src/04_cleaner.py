import numpy as np
import pandas as pd
from typing import List

class MotifCleaner:
    """
    Cleans extracted time-series motifs by applying a sequence of strict filters.
    Expects a DataFrame with columns like 'Signal_ID', 'Start_Index', 'End_Index', 
    'Length', 'Maximum', 'Minimum', 'Energy', and 'Raw_Signal'.
    """
    
    def __init__(self, data_path: str):
        # We work on a copy to preserve the original extraction output
        self.df = pd.read_csv(data_path)
        import ast        
        # Convert the string representation of lists to actual numpy arrays globally
        if isinstance(self.df['Raw_Signal'].iloc[0], str):
             print("Converting 'Raw_Signal' strings to arrays...")
             self.df['Raw_Signal'] = self.df['Raw_Signal'].apply(ast.literal_eval)
        self.initial_count = len(self.df)

        print(f"Initialized MotifCleaner with {self.initial_count} motifs.")

    def remove_length_anomalies(self, min_len: int = 10, max_len: int = 500) -> 'MotifCleaner':
        """Removes motifs that are excessively short or long."""
        before = len(self.df)
        self.df = self.df[(self.df['Length'] >= min_len) & (self.df['Length'] <= max_len)]
        dropped = before - len(self.df)
        print(f"[-] Length anomalies removed: {dropped} (Criteria: {min_len} <= len <= {max_len})")
        return self

    def remove_duplicates(self) -> 'MotifCleaner':
        """
        Removes exact duplicate motifs based on Signal_ID and spatial indices.
        This handles overlaps if multiple extraction methods were combined.
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Signal_ID', 'Start_Index', 'End_Index'])
        dropped = before - len(self.df)
        print(f"[-] Duplicated motifs removed: {dropped}")
        return self

    def remove_constant_motifs(self, std_threshold: float = 1e-4) -> 'MotifCleaner':
        """
        Removes motifs where the signal does not change (e.g., sensor froze).
        Measured by evaluating the standard deviation of the raw signal.
        """
        before = len(self.df)
        
        def is_constant(signal_list):
            return np.std(signal_list) < std_threshold
            
        # Keep rows where it is NOT constant
        mask = ~self.df['Raw_Signal'].apply(is_constant)
        self.df = self.df[mask]
        
        dropped = before - len(self.df)
        print(f"[-] Constant motifs removed: {dropped} (Std Dev < {std_threshold})")
        return self

    def remove_flat_motifs(self, amplitude_threshold: float = 1.0) -> 'MotifCleaner':
        """
        Removes flat motifs where the difference between max and min is negligible.
        Unlike constant motifs, these might have slight noise but lack a true 'shape'.
        """
        before = len(self.df)
        # Using the pre-calculated features for speed
        amplitude_diff = self.df['Maximum'] - self.df['Minimum']
        self.df = self.df[amplitude_diff >= amplitude_threshold]
        
        dropped = before - len(self.df)
        print(f"[-] Flat motifs removed: {dropped} (Max - Min < {amplitude_threshold})")
        return self

    def remove_outliers(self, columns: List[str] = ['Energy', 'Length'], k: float = 1.5) -> 'MotifCleaner':
        """
        Removes statistical outliers using the Interquartile Range (IQR) method 
        on specified numeric features.
        """
        before = len(self.df)
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (k * IQR)
                upper_bound = Q3 + (k * IQR)
                
                # Update mask to only keep items within bounds for this column
                col_mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                mask = mask & col_mask
                
        self.df = self.df[mask]
        dropped = before - len(self.df)
        print(f"[-] Statistical outliers removed: {dropped} (Based on IQR of {columns})")
        return self

    def get_clean_data(self) -> pd.DataFrame:
        """Returns the cleaned DataFrame and prints the final summary."""
        final_count = len(self.df)
        retention = (final_count / self.initial_count) * 100 if self.initial_count > 0 else 0
        print(f"\n[!] Cleaning Complete. Retained {final_count} motifs ({retention:.1f}% retention).")
        return self.df.copy()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":

    # 2. Run the cleaning pipeline
    cleaner = MotifCleaner('data/raw_motifs.csv')
    
    # Method chaining for clean, readable pipeline execution
    cleaned_df = (
        cleaner
        .remove_length_anomalies(min_len=10, max_len=200)
        .remove_duplicates()
        .remove_constant_motifs(std_threshold=1e-4)
        .remove_flat_motifs(amplitude_threshold=1.0)
        .remove_outliers(columns=['Energy', 'Length'], k=1.5)
        .get_clean_data()
    )

    cleaned_df.to_csv("data/cleaned_motifs.csv", index=False)