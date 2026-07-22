import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import pywt
from statsmodels.tsa.stattools import acf, pacf

class MotifFeatureEngineer:
    """
    Extracts a comprehensive set of features (Statistical, Shape, Frequency, 
    Wavelet, and Time-Series) from the raw time-series arrays of cleaned motifs.
    """
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        import ast        
        # Convert the string representation of lists to actual numpy arrays globally
        if isinstance(self.df['Raw_Signal'].iloc[0], str):
             print("Converting 'Raw_Signal' strings to arrays...")
             self.df['Raw_Signal'] = self.df['Raw_Signal'].apply(ast.literal_eval)
        # Raw_Signal is a numpy array for vectorized operations
        if 'Raw_Signal' in self.df.columns:
            self.df['Raw_Signal'] = self.df['Raw_Signal'].apply(np.array)
        else:
            raise ValueError("Input DataFrame must contain a 'Raw_Signal' column.")

    # ==========================================
    # 1. STATISTICAL FEATURES
    # ==========================================
    @staticmethod
    def _extract_statistical(signal: np.ndarray) -> dict:
        return {
            'Stat_Mean': np.mean(signal),
            'Stat_Median': np.median(signal),
            'Stat_Variance': np.var(signal),
            'Stat_Std': np.std(signal),
            'Stat_Max': np.max(signal),
            'Stat_Min': np.min(signal),
            'Stat_Range': np.ptp(signal), # Peak-to-peak (Max - Min)
            'Stat_Skewness': skew(signal),
            'Stat_Kurtosis': kurtosis(signal)
        }

    # ==========================================
    # 2. SHAPE FEATURES
    # ==========================================
    @staticmethod
    def _extract_shape(signal: np.ndarray) -> dict:
        # Peak analysis
        peaks, _ = find_peaks(signal)
        peak_count = len(peaks)
        peak_distance = np.mean(np.diff(peaks)) if peak_count > 1 else 0
        
        # Slopes
        differences = np.diff(signal)
        avg_slope = np.mean(differences)
        
        # Rise and Fall times (simplified as proportion of positive vs negative slopes)
        positive_slopes = differences[differences > 0]
        negative_slopes = differences[differences < 0]
        rise_time = len(positive_slopes) / len(signal)
        fall_time = len(negative_slopes) / len(signal)
        
        # Zero Crossings (shifted to mean 0 first to find true crossings of the baseline)
        centered_signal = signal - np.mean(signal)
        zero_crossings = np.where(np.diff(np.sign(centered_signal)))[0]
        
        return {
            'Shape_Length': len(signal),
            'Shape_Peak_Count': peak_count,
            'Shape_Avg_Peak_Dist': peak_distance,
            'Shape_Avg_Slope': avg_slope,
            'Shape_Rise_Time_Ratio': rise_time,
            'Shape_Fall_Time_Ratio': fall_time,
            'Shape_Area': np.trapezoid(signal), # Trapezoidal rule for area
            'Shape_Energy': np.sum(signal ** 2),
            'Shape_Zero_Crossings': len(zero_crossings)
        }

    # ==========================================
    # 3. FREQUENCY FEATURES (FFT)
    # ==========================================
    @staticmethod
    def _extract_frequency(signal: np.ndarray) -> dict:
        n = len(signal)
        # FFT and frequencies
        yf = fft(signal)
        xf = fftfreq(n)
        
        # Positive frequencies only
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        power_spectrum = np.abs(yf[pos_mask]) ** 2
        
        # Dominant frequency
        if len(power_spectrum) > 0:
            dominant_idx = np.argmax(power_spectrum)
            dominant_freq = xf_pos[dominant_idx]
            
            # Spectral Entropy (normalize power spectrum to sum to 1)
            ps_norm = power_spectrum / np.sum(power_spectrum)
            spec_entropy = entropy(ps_norm)
        else:
            dominant_freq = 0
            spec_entropy = 0
            
        return {
            'Freq_Dominant': dominant_freq,
            'Freq_Spectral_Entropy': spec_entropy,
            'Freq_Mean_Power': np.mean(power_spectrum) if len(power_spectrum) > 0 else 0
        }

    # ==========================================
    # 4. WAVELET FEATURES (DWT)
    # ==========================================
    @staticmethod
    def _extract_wavelet(signal: np.ndarray, wavelet_type='db4') -> dict:
        try:
            # Discrete Wavelet Transform
            cA, cD = pywt.dwt(signal, wavelet_type)
            return {
                'Wavelet_Approx_Mean': np.mean(cA),
                'Wavelet_Approx_Std': np.std(cA),
                'Wavelet_Approx_Energy': np.sum(cA ** 2),
                'Wavelet_Detail_Mean': np.mean(cD),
                'Wavelet_Detail_Std': np.std(cD),
                'Wavelet_Detail_Energy': np.sum(cD ** 2)
            }
        except Exception:
            # Fallback if signal is too short for the chosen wavelet
            return {
                'Wavelet_Approx_Mean': 0, 'Wavelet_Approx_Std': 0, 'Wavelet_Approx_Energy': 0,
                'Wavelet_Detail_Mean': 0, 'Wavelet_Detail_Std': 0, 'Wavelet_Detail_Energy': 0
            }

    # ==========================================
    # 5. TIME-SERIES FEATURES (ACF / PACF)
    # ==========================================
    @staticmethod
    def _extract_timeseries(signal: np.ndarray) -> dict:
        # Autocorrelation (Lag 1 and Lag 2)
        try:
            # nlags determines how many lags to compute. We need at least 2.
            lags = min(10, len(signal) - 1)
            acf_vals = acf(signal, nlags=lags, fft=True)
            lag_1_acf = acf_vals[1] if lags >= 1 else 0
            lag_2_acf = acf_vals[2] if lags >= 2 else 0
        except Exception:
            lag_1_acf, lag_2_acf = 0, 0
            
        # PACF (Requires signal length > nlags)
        try:
            pacf_lags = min(5, len(signal) // 2 - 1)
            if pacf_lags > 1:
                pacf_vals = pacf(signal, nlags=pacf_lags, method='ywm')
                lag_1_pacf = pacf_vals[1]
            else:
                lag_1_pacf = 0
        except Exception:
            lag_1_pacf = 0

        return {
            'TS_ACF_Lag1': lag_1_acf,
            'TS_ACF_Lag2': lag_2_acf,
            'TS_PACF_Lag1': lag_1_pacf
        }

    # ==========================================
    # MAIN EXTRACTION PIPELINE
    # ==========================================
    def generate_features(self) -> pd.DataFrame:
        """
        Iterates over all motifs and extracts the full feature suite.
        Returns a new DataFrame with the original metadata and all computed features.
        """
        print(f"Extracting features for {len(self.df)} motifs...")
        all_features = []
        
        for idx, row in self.df.iterrows():
            signal = row['Raw_Signal']
            
            # If a motif somehow survived cleaning but is too short, handle it
            if len(signal) < 3:
                continue 
                
            # Aggregate all feature dictionaries
            features = {
                'Motif_ID': row.get('Motif_ID', f"motif_{idx}"),
                'Signal_ID': row.get('Signal_ID', 'unknown'),
            }
            
            features.update(self._extract_statistical(signal))
            features.update(self._extract_shape(signal))
            features.update(self._extract_frequency(signal))
            features.update(self._extract_wavelet(signal))
            features.update(self._extract_timeseries(signal))
            
            all_features.append(features)
            
        features_df = pd.DataFrame(all_features)
        
        # Merge back with any other original metadata (like Start/End index) if desired,
        # but usually, the feature matrix is kept numerical for clustering.
        print(f"Extraction complete! Feature matrix shape: {features_df.shape}")
        return features_df


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure installed dependencies:
    # pip install numpy pandas scipy pywt statsmodels

    # 2. Run Feature Engineering
    engineer = MotifFeatureEngineer('data/cleaned_motifs.csv')
    feature_matrix = engineer.generate_features()
    feature_matrix.to_csv("data/future_motifs.csv", index=False)
    
    # 3. View the generated features
    print("\nSample Feature Output (First Motif):")
    first_motif_features = feature_matrix.iloc[0].to_dict()
    for key, value in list(first_motif_features.items())[:15]: # Show first 15 for brevity
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    print("...\n(Total features extracted per motif:", feature_matrix.shape[1] - 2, ")")