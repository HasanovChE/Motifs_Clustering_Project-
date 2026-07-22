import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer
)

class SignalPreprocessor:
    """
    A comprehensive pipeline for time-series signal preprocessing.
    Handles Noise Removal, Baseline Removal, and Normalization.
    """
    def __init__(self, filepath: str):
        # Load data, first row is header and columns are individual signals
        print(f"Loading data from {filepath}...")
        self.data = pd.read_csv(filepath)
        self.original_signals = self.data.copy()
        print(f"Data loaded successfully. Shape: {self.data.shape}")

    # ==========================================
    # 1. NOISE REMOVAL METHODS
    # ==========================================
    @staticmethod
    def apply_moving_average(signal, window_size=5):
        return pd.Series(signal).rolling(window=window_size, min_periods=1, center=True).mean().values

    @staticmethod
    def apply_savgol_filter(signal, window_length=11, polyorder=3):
        # Window length is odd and less than signal length
        window_length = min(window_length, len(signal) if len(signal) % 2 != 0 else len(signal) - 1)
        return savgol_filter(signal, window_length, polyorder)

    @staticmethod
    def apply_median_filter(signal, kernel_size=5):
        return medfilt(signal, kernel_size=kernel_size)

    @staticmethod
    def apply_gaussian_filter(signal, sigma=2.0):
        return gaussian_filter1d(signal, sigma=sigma)

    @staticmethod
    def apply_low_pass_filter(signal, cutoff=0.1, fs=1.0, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    # ==========================================
    # 2. BASELINE REMOVAL METHODS
    # ==========================================
    @staticmethod
    def remove_hardcoded_baseline(signal, baseline_value=120.0):
        """
        Subtracts the known steady-state baseline (120) from the signal.
        This shifts the entire signal so the noise floor sits at 0.
        """
        return signal - baseline_value

    @staticmethod
    def remove_baseline_simple(signal):
        """Subtracts the median of the signal as a simple baseline."""
        return signal - np.median(signal)

    @staticmethod
    def remove_baseline_polynomial(signal, degree=3):
        """Fits a polynomial to the signal and subtracts it."""
        x = np.arange(len(signal))
        coefs = np.polyfit(x, signal, degree)
        baseline = np.polyval(coefs, x)
        return signal - baseline

    @staticmethod
    def remove_baseline_rolling(signal, window_size=50):
        """Subtracts a large-window rolling mean."""
        baseline = pd.Series(signal).rolling(window=window_size, min_periods=1, center=True).mean().values
        return signal - baseline

    # ==========================================
    # 3. NORMALIZATION METHODS
    # ==========================================
    @staticmethod
    def normalize_minmax(signal):
        scaler = MinMaxScaler()
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    @staticmethod
    def normalize_zscore(signal):
        scaler = StandardScaler()
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    @staticmethod
    def normalize_robust(signal):
        scaler = RobustScaler()
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    @staticmethod
    def normalize_quantile(signal):
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(signal), 1000))
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    @staticmethod
    def normalize_power(signal):
        scaler = PowerTransformer(method='yeo-johnson')
        return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    # ==========================================
    # VISUALIZATION
    # ==========================================
    def compare_preprocessing(self, column_index=0):
        """
        Visualizes the original signal against all preprocessing techniques.
        Per the requirements, plots are laid out vertically: y-axis is index, x-axis is signal value.
        """
        # Get the first signal for demonstration
        col_name = self.data.columns[column_index]
        raw_signal = self.data[col_name].values
        idx = np.arange(len(raw_signal))

        # Generate processed variants
        # 1. Noise
        sig_ma = self.apply_moving_average(raw_signal)
        sig_sg = self.apply_savgol_filter(raw_signal)
        sig_med = self.apply_median_filter(raw_signal)
        sig_gauss = self.apply_gaussian_filter(raw_signal)
        sig_lp = self.apply_low_pass_filter(raw_signal)

        # 2. Baseline
        sig_base_simp = self.remove_baseline_simple(raw_signal)
        sig_base_poly = self.remove_baseline_polynomial(raw_signal)
        sig_base_roll = self.remove_baseline_rolling(raw_signal)

        # 3. Normalization (applied to baseline-removed & filtered signal to show full pipeline)
        # Savitzky-Golay + Polynomial Baseline signal for normalization tests
        clean_sig = self.remove_baseline_polynomial(self.apply_savgol_filter(raw_signal))
        
        sig_minmax = self.normalize_minmax(clean_sig)
        sig_zscore = self.normalize_zscore(clean_sig)
        sig_robust = self.normalize_robust(clean_sig)
        sig_quant = self.normalize_quantile(clean_sig)
        sig_power = self.normalize_power(clean_sig)

        # Setup plotting framework
        fig, axes = plt.subplots(3, 5, figsize=(30, 24), sharey=True)
        fig.suptitle(f"Preprocessing Comparison for Signal: {col_name}\n(Y-axis = Index, X-axis = Amplitude)", fontsize=18)

        def plot_vertical(ax, x_data, title, color):
            ax.plot(x_data, idx, color=color, linewidth=1)
            ax.set_title(title)
            ax.invert_yaxis() # Index 0 at the top, like depth logs
            ax.grid(True, linestyle='--', alpha=0.6)

        # Row 1: Noise Removal
        plot_vertical(axes[0,0], raw_signal, "Original", 'black')
        plot_vertical(axes[0,1], sig_ma, "Moving Average", 'blue')
        plot_vertical(axes[0,2], sig_sg, "Savitzky-Golay", 'orange')
        plot_vertical(axes[0,3], sig_gauss, "Gaussian Filter", 'green')
        plot_vertical(axes[0,4], sig_lp, "Low-pass Filter", 'red')

        # Row 2: Baseline Removal
        plot_vertical(axes[1,0], raw_signal, "Original", 'black')
        plot_vertical(axes[1,1], sig_base_simp, "Subtract Median", 'purple')
        plot_vertical(axes[1,2], sig_base_poly, "Polynomial Fit", 'brown')
        plot_vertical(axes[1,3], sig_base_roll, "Rolling Mean", 'cyan')
        axes[1,4].axis('off') # Empty subplot

        # Row 3: Normalization (Applied on cleaned data)
        plot_vertical(axes[2,0], clean_sig, "Cleaned (SG + Poly)", 'black')
        plot_vertical(axes[2,1], sig_minmax, "MinMax Scaler", 'magenta')
        plot_vertical(axes[2,2], sig_zscore, "Z-Score (Standard)", 'olive')
        plot_vertical(axes[2,3], sig_robust, "Robust Scaler", 'teal')
        plot_vertical(axes[2,4], sig_power, "Power Transform", 'navy')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    try:        
        # 1. Initialize the processor
        processor = SignalPreprocessor("data/signals_data.csv")
        
        # 2. Create an empty DataFrame to store all cleaned signals
        processed_signals = pd.DataFrame()
        
        # 3. Apply filters to each column/signal (e.g., Savitzky-Golay + Polynomial Baseline)
        print("Cleaning signals, please wait...")
        for col in processor.data.columns:
            raw_sig = processor.data[col].values
            
            # First remove noise, then adjust the baseline
            clean_sig = processor.remove_baseline_polynomial(
                processor.apply_savgol_filter(raw_sig)
            )
            processed_signals[col] = clean_sig
            
        # 4. Save the cleaned data to a CSV file
        processed_signals.to_csv("data/preprocessed_signals_data.csv", index=False)
        print("Cleaned data successfully saved as 'data/preprocessed_signals_data.csv'!")
        
        # The visualization plot:
        processor.compare_preprocessing(column_index=0)
        
    except FileNotFoundError:
        print("ERROR: 'signals_data.csv' not found. Please ensure the file is in the same directory.")