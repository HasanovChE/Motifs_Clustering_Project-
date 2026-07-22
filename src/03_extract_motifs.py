import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson

class MotifExtractor():
    """
    Extracts motifs from time-series signals using multiple detection methods.
    Each extracted motif is stored with its statistical metadata and raw signal segment.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data: A pandas DataFrame where each column is an individual signal.
        """
        
        self.data = pd.read_csv(data_path) 
        self.motifs_database = []

    def _calculate_features(self, signal_col_name: str, motif_counter: int, 
                            segment: np.ndarray, start_idx: int, end_idx: int) -> dict:
        """
        Helper method to compute the required metrics for each extracted motif.
        """
        # Motif naming convention: e.g., signal_1.1, signal_1.2
        # If the column name is '1', it becomes 'signal_1.1'
        clean_name = str(signal_col_name).replace("signal_", "")
        motif_id = f"signal_{clean_name}.{motif_counter}"
        
        return {
            "Motif_ID": motif_id,
            "Signal_ID": signal_col_name,
            "Start_Index": start_idx,
            "End_Index": end_idx,
            "Length": len(segment),
            "Maximum": np.max(segment),
            "Minimum": np.min(segment),
            "Mean": np.mean(segment),
            "Area": simpson(segment), # Using Simpson's rule for area under the curve
            "Energy": np.sum(np.square(segment)), # Sum of squared amplitudes
            "Raw_Signal": segment.tolist()
        }

    def _extract_from_continuous_regions(self, signal: np.ndarray, mask: np.ndarray, 
                                         signal_name: str, min_length: int = 5) -> list:
        """
        Helper method to extract segments where a boolean mask is True.
        """
        motifs = []
        # Find indices where mask changes from False to True or True to False
        padded_mask = np.concatenate(([False], mask, [False]))
        changes = np.where(padded_mask[:-1] != padded_mask[1:])[0]
        
        # Reshape into start and end pairs
        regions = changes.reshape(-1, 2)
        
        motif_counter = 1
        for start, end in regions:
            if (end - start) >= min_length:
                segment = signal[start:end]
                features = self._calculate_features(signal_name, motif_counter, segment, start, end)
                motifs.append(features)
                motif_counter += 1
                
        return motifs

    # ==========================================
    # METHOD 1: Fixed Threshold (110)
    # ==========================================
    def extract_fixed_threshold(self, threshold: float = 110.0, min_length: int = 5):
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            mask = signal > threshold
            motifs = self._extract_from_continuous_regions(signal, mask, col, min_length)
            extracted.extend(motifs)
        return pd.DataFrame(extracted)

    # ==========================================
    # METHOD 2: Adaptive Threshold (Mean + k * Std)
    # ==========================================
    def extract_adaptive_threshold(self, k: float = 1.5, min_length: int = 5):
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            threshold = np.mean(signal) + (k * np.std(signal))
            mask = signal > threshold
            motifs = self._extract_from_continuous_regions(signal, mask, col, min_length)
            extracted.extend(motifs)
        return pd.DataFrame(extracted)

    # ==========================================
    # METHOD 3: Percentile Threshold
    # ==========================================
    def extract_percentile_threshold(self, percentile: float = 95.0, min_length: int = 5):
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            threshold = np.percentile(signal, percentile)
            mask = signal > threshold
            motifs = self._extract_from_continuous_regions(signal, mask, col, min_length)
            extracted.extend(motifs)
        return pd.DataFrame(extracted)

    # ==========================================
    # METHOD 4: Peak Detection (scipy.signal)
    # ==========================================
    def extract_peak_detection(self, distance: int = 10, prominence: float = None):
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            
            # Find peaks
            peaks, properties = find_peaks(signal, distance=distance, prominence=prominence)
            
            # Calculate widths to define the start and end of the peak motif
            widths_data = peak_widths(signal, peaks, rel_height=0.5) # Width at half prominence
            
            # The widths_data tuple contains: widths, width_heights, left_ips, right_ips
            left_ips = np.floor(widths_data[2]).astype(int)
            right_ips = np.ceil(widths_data[3]).astype(int)
            
            motif_counter = 1
            for start, end in zip(left_ips, right_ips):
                if end > start:
                    segment = signal[start:end]
                    features = self._calculate_features(col, motif_counter, segment, start, end)
                    extracted.append(features)
                    motif_counter += 1
                    
        return pd.DataFrame(extracted)

    # ==========================================
    # METHOD 5: Derivative Based (Slope changes)
    # ==========================================
    def extract_derivative_based(self, slope_threshold: float = 0.5, min_length: int = 5):
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            
            # First derivative (slope)
            derivative = np.gradient(signal)
            
            # Define motif regions where the absolute rate of change is significant
            mask = np.abs(derivative) > slope_threshold
            motifs = self._extract_from_continuous_regions(signal, mask, col, min_length)
            extracted.extend(motifs)

        return pd.DataFrame(extracted)


    
    def extract_from_zero_baseline(self, amplitude_threshold: float = 10.0, min_length: int = 15):
        """
        Extracts motifs from preprocessed signals where the baseline has been shifted to 0.
        It triggers when the signal spikes up or dips down beyond the amplitude_threshold.
        """
        extracted = []
        for col in self.data.columns:
            signal = self.data[col].values
            
            # Since preprocessing centered the steady state at 0, 
            # we look for absolute deviations (e.g., > 10 or < -10)
            mask = np.abs(signal) >= amplitude_threshold
            
            # Extract regions where the mask is True continuously
            motifs = self._extract_from_continuous_regions(signal, mask, col, min_length)
            extracted.extend(motifs)
            
        return pd.DataFrame(extracted)


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    
    extractor = MotifExtractor('data/preprocessed_signals_data.csv')

    
    print("Extracting via Fixed Threshold (110)...")
    df_fixed = extractor.extract_fixed_threshold(threshold=110)
    print(f"Found {len(df_fixed)} motifs.")
    
    print("Extracting via Adaptive Threshold (Mean + 1.5*Std)...")
    df_adaptive = extractor.extract_adaptive_threshold(k=1.5)
    print(f"Found {len(df_adaptive)} motifs.")

    print("Extracting via Percentile Threshold (95th Percentile)...")
    df_percentile = extractor.extract_percentile_threshold(percentile=95)
    print(f"Found {len(df_percentile)} motifs.")
    
    print("Extracting via Peak Detection...")
    df_peaks = extractor.extract_peak_detection(distance=15)
    print(f"Found {len(df_peaks)} motifs.")

    print("Extracting via Derivative-Based Method...")
    df_derivative = extractor.extract_derivative_based(slope_threshold=0.5)
    print(f"Found {len(df_derivative)} motifs.")
    
    # Add a column to identify the extraction method
    df_fixed["extraction_method"] = "fixed_threshold"
    df_adaptive["extraction_method"] = "adaptive_threshold"
    df_peaks["extraction_method"] = "peak_detection"
    df_percentile["extraction_method"] = "percentile_threshold"
    df_derivative["extraction_method"] = "derivative_based"
    
    # Combine all dataframes vertically
    df_combined = pd.concat([df_fixed, df_adaptive, df_peaks,df_percentile,df_derivative], ignore_index=True)
    
    # Save as CSV
    df_combined.to_csv("data/extracted_motifs.csv", index=False)
    
    print(f"Combined dataframe created with {len(df_combined)} total rows and saved.")
    
    # Example of how the output looks:
    if not df_fixed.empty:
        # Dropping the Raw_Signal column just for printing cleanly to the terminal
        print("\nSample Output (First 3 motifs, Fixed Threshold):")
        print(df_fixed.drop(columns=['Raw_Signal']).head(3).to_string())
    
    raw_motifs = extractor.extract_from_zero_baseline()
    raw_motifs.to_csv("data/raw_motifs.csv", index=False)
    print("##################################################################################")
    print(f"Combined dataframe created with {len(raw_motifs)} total rows and saved.")