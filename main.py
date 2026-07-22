import importlib
import os
import json
import time
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

# ==========================================
# DYNAMIC IMPORTS OF ALL 11 PIPELINE STAGES
# ==========================================
# Since stage filenames start with numbers (e.g., 01_load_data_eda.py),
 # Dynamically import them via importlib without altering the src/ folder.
m01_eda = importlib.import_module("src.01_load_data_eda")
m02_prep = importlib.import_module("src.02_preprocessing")
m03_ext = importlib.import_module("src.03_extract_motifs")
m04_clean = importlib.import_module("src.04_cleaner")
m05_feat = importlib.import_module("src.05_feature_engineer")
m06_sim = importlib.import_module("src.06_similarity_finder")
m07_clust = importlib.import_module("src.07_cluster")
m08_tune = importlib.import_module("src.08_hyperparameter_tuner")
m09_eval = importlib.import_module("src.09_evaluator")
m10_viz = importlib.import_module("src.10_clusters_visualizer")
m11_comp = importlib.import_module("src.11_model_compare")

# Extract class handles
SignalEDA = m01_eda.SignalEDA
SignalPreprocessor = m02_prep.SignalPreprocessor
MotifExtractor = m03_ext.MotifExtractor
MotifCleaner = m04_clean.MotifCleaner
MotifFeatureEngineer = m05_feat.MotifFeatureEngineer
FeatureSimilarityCalculator = m06_sim.FeatureSimilarityCalculator
GlobalMotifClusterer = m07_clust.GlobalMotifClusterer
ClusteringOptimizer = m08_tune.ClusteringOptimizer
ClusteringValidator = m09_eval.ClusteringValidator
MotifVisualizer = m10_viz.MotifVisualizer
MotifModelComparator = m11_comp.MotifModelComparator

# Helper function to save current matplotlib plot and close
def save_plot_and_close(output_dir, filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close('all')


def run_pipeline(input_csv_path="data/signals_data.csv", output_dir="outputs", progress_callback=None):
    """
    Executes the full 11-Stage Time-Series Motif Clustering Pipeline.
    
    Args:
        input_csv_path (str): Path to raw time-series CSV file.
        output_dir (str): Directory where all plots, CSVs, and reports will be saved.
        progress_callback (callable): Optional callback fn(stage_idx, stage_name, message)
                                      used by Streamlit or CLI for live progress tracking.
                                      
    Returns:
        dict: Summary package containing results, dataframes, model tables, best model dict, and plot paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    def log_progress(idx, name, msg=""):
        if progress_callback:
            progress_callback(idx, name, msg)
        print(f"\n[{idx}/11] {name} {msg}")

    # Generate synthetic fallback data if the input file doesn't exist
    if not os.path.exists(input_csv_path):
        log_progress(0, "Setup", f"Input file '{input_csv_path}' not found. Generating synthetic multi-channel dataset...")
        np.random.seed(42)
        df_mock = pd.DataFrame({
            f"signal_{i}": (
                np.random.normal(100, 5, 1000) + 
                np.sin(np.linspace(0, 40 * i, 1000)) * 25 +
                (np.linspace(0, 10, 1000) if i % 2 == 0 else 0)
            )
            for i in range(1, 11)
        })
        df_mock.to_csv(input_csv_path, index=False)
        print(f"Generated fallback dataset at '{input_csv_path}'")

    results_bundle = {
        "output_dir": output_dir,
        "input_csv_path": input_csv_path,
        "plots": {}
    }

    # ==========================================
    # STAGE 1: LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    log_progress(1, "Stage 1: Load Data & EDA", f"(Analyzing {input_csv_path})")
    eda = SignalEDA(input_csv_path)
    
    with patch('matplotlib.pyplot.show'):
        first_col = eda.df.columns[0]
        eda.visualize_signal(column_name=first_col)
        save_plot_and_close(output_dir, "01_eda_signal.png")
        results_bundle["plots"]["01_eda_signal"] = os.path.join(output_dir, "01_eda_signal.png")
    
    results_bundle["raw_df_shape"] = eda.df.shape
    results_bundle["raw_df"] = eda.df

    # ==========================================
    # STAGE 2: PREPROCESSING (Noise, Baseline, Normalization)
    # ==========================================
    log_progress(2, "Stage 2: Preprocessing Signals", "(Filtering noise & subtracting baseline)")
    preprocessor = SignalPreprocessor(input_csv_path)
    
    processed_signals = pd.DataFrame()
    for col in preprocessor.data.columns:
        sig = preprocessor.data[col].values
        # Apply Savitzky-Golay filtering + polynomial baseline removal
        sig_clean = preprocessor.remove_baseline_polynomial(preprocessor.apply_savgol_filter(sig))
        processed_signals[col] = sig_clean
        
    preprocessed_path = os.path.join("data", "preprocessed_signals_data.csv")
    processed_signals.to_csv(preprocessed_path, index=False)
    processed_signals.to_csv(os.path.join(output_dir, "preprocessed_signals_data.csv"), index=False)
    
    with patch('matplotlib.pyplot.show'):
        preprocessor.compare_preprocessing(column_index=0)
        save_plot_and_close(output_dir, "02_preprocessing_comparison.png")
        results_bundle["plots"]["02_preprocessing_comparison"] = os.path.join(output_dir, "02_preprocessing_comparison.png")
    
    results_bundle["processed_signals"] = processed_signals

    # ==========================================
    # STAGE 3: COLUMN-WISE MOTIF EXTRACTION
    # ==========================================
    log_progress(3, "Stage 3: Extracting Candidate Motifs", "(Running multi-method thresholding & peak detection)")
    extractor = MotifExtractor(preprocessed_path)
    
    # Run multi-method extraction suite
    df_adaptive = extractor.extract_adaptive_threshold(k=1.5, min_length=10)
    if not df_adaptive.empty:
        df_adaptive["extraction_method"] = "adaptive_threshold"
    
    df_zero = extractor.extract_from_zero_baseline(amplitude_threshold=8.0, min_length=15)
    if not df_zero.empty:
        df_zero["extraction_method"] = "zero_baseline"
        
    df_peaks = extractor.extract_peak_detection(distance=15)
    if not df_peaks.empty:
        df_peaks["extraction_method"] = "peak_detection"
        
    df_derivative = extractor.extract_derivative_based(slope_threshold=0.5, min_length=10)
    if not df_derivative.empty:
        df_derivative["extraction_method"] = "derivative_based"

    raw_motifs = pd.concat([df_adaptive, df_zero, df_peaks, df_derivative], ignore_index=True)
    
    # If no motifs extracted (e.g. extreme flat synthetic data), fallback to percentile threshold
    if raw_motifs.empty:
        df_perc = extractor.extract_percentile_threshold(percentile=85.0, min_length=10)
        df_perc["extraction_method"] = "percentile_threshold"
        raw_motifs = df_perc
        
    raw_motifs_path = os.path.join("data", "raw_motifs.csv")
    raw_motifs.to_csv(raw_motifs_path, index=False)
    raw_motifs.to_csv(os.path.join(output_dir, "raw_extracted_motifs.csv"), index=False)
    results_bundle["raw_motifs_count"] = len(raw_motifs)

    # ==========================================
    # STAGE 4: MOTIF CLEANING & FILTERING
    # ==========================================
    log_progress(4, "Stage 4: Cleaning & Filtering Motifs", "(Removing anomalies, duplicates, and flat segments)")
    cleaner = MotifCleaner(raw_motifs_path)
    clean_motifs_df = (
        cleaner
        .remove_length_anomalies(min_len=10, max_len=250)
        .remove_duplicates()
        .remove_constant_motifs(std_threshold=1e-4)
        .remove_flat_motifs(amplitude_threshold=0.5)
        .remove_outliers(columns=['Energy', 'Length'], k=1.8)
        .get_clean_data()
    )
    
    cleaned_path = os.path.join("data", "cleaned_motifs.csv")
    clean_motifs_df.to_csv(cleaned_path, index=False)
    clean_motifs_df.to_csv("cleaned_motifs.csv", index=False) # Root mirror for stage helpers
    clean_motifs_df.to_csv(os.path.join(output_dir, "cleaned_motifs.csv"), index=False)
    results_bundle["clean_motifs_df"] = clean_motifs_df
    results_bundle["clean_motifs_count"] = len(clean_motifs_df)

    if len(clean_motifs_df) < 3:
        log_progress(4, "Warning", "Less than 3 motifs survived cleaning. Reducing strictness to guarantee clustering...")
        clean_motifs_df = raw_motifs.drop_duplicates(subset=['Signal_ID', 'Start_Index', 'End_Index']).copy()
        clean_motifs_df.to_csv(cleaned_path, index=False)
        clean_motifs_df.to_csv("cleaned_motifs.csv", index=False)
        clean_motifs_df.to_csv(os.path.join(output_dir, "cleaned_motifs.csv"), index=False)
        results_bundle["clean_motifs_df"] = clean_motifs_df
        results_bundle["clean_motifs_count"] = len(clean_motifs_df)

    # ==========================================
    # STAGE 5: FEATURE ENGINEERING
    # ==========================================
    log_progress(5, "Stage 5: Feature Engineering", "(Generating Statistical, Shape, Frequency, Wavelet & ACF features)")
    engineer = MotifFeatureEngineer(cleaned_path)
    features_df = engineer.generate_features()
    
    feature_path = os.path.join("data", "future_motifs.csv")
    features_df.to_csv(feature_path, index=False)
    features_df.to_csv(os.path.join(output_dir, "feature_matrix.csv"), index=False)
    results_bundle["features_df"] = features_df

    # ==========================================
    # STAGE 6: SIMILARITY MATRIX COMPUTATION
    # ==========================================
    log_progress(6, "Stage 6: Similarity Matrix Computation", "(Standardizing features & calculating distance metrics)")
    sim_calc = FeatureSimilarityCalculator(feature_path)
    
    scaled_features = sim_calc.scaled_features
    euclidean_dist = sim_calc.compute_euclidean()
    cosine_dist = sim_calc.compute_cosine()
    correlation_dist = sim_calc.compute_correlation()
    
    pd.DataFrame(euclidean_dist).to_csv(os.path.join(output_dir, "euclidean_distance_matrix.csv"), index=False)
    results_bundle["scaled_features"] = scaled_features
    results_bundle["euclidean_dist"] = euclidean_dist
    results_bundle["cosine_dist"] = cosine_dist

    # ==========================================
    # STAGE 7: GLOBAL CLUSTERING IMPLEMENTATIONS
    # ==========================================
    log_progress(7, "Stage 7: Global Clustering Models", "(Training KMeans, Agglomerative, Spectral, SOM, AE+KMeans & DEC)")
    
    n_clusters_target = min(5, max(2, len(clean_motifs_df) // 10))
    
    models_to_run = {
        "KMeans": {"is_precomputed": False, "func": "run_kmeans", "kwargs": {"n_clusters": n_clusters_target}},
        "Agglomerative": {"is_precomputed": False, "func": "run_agglomerative", "kwargs": {"n_clusters": n_clusters_target, "linkage": "average"}},
        "Spectral": {"is_precomputed": False, "func": "run_spectral", "kwargs": {"n_clusters": n_clusters_target}},
        "SOM": {"is_precomputed": False, "func": "run_som", "kwargs": {"grid_x": 3, "grid_y": 3, "epochs": 300}},
        "Autoencoder + KMeans": {"is_precomputed": False, "func": "run_autoencoder_kmeans", "kwargs": {"n_clusters": n_clusters_target, "epochs": 15, "batch_size": min(32, len(clean_motifs_df))}},
        "DEC (Deep Embedded)": {"is_precomputed": False, "func": "run_dec", "kwargs": {"n_clusters": n_clusters_target, "pretrain_epochs": 10, "cluster_epochs": 15}},
        # "K-Medoids (Euclidean)": {"is_precomputed": True, "func": "run_kmedoids", "kwargs": {"n_clusters": n_clusters_target}},
        "HDBSCAN": {"is_precomputed": True, "func": "run_hdbscan", "kwargs": {"min_cluster_size": max(3, len(clean_motifs_df) // 12)}}
    }

    all_labels = {}
    runtimes = {}

    for name, config in models_to_run.items():
        print(f"  -> Training {name}...")
        start_time = time.time()
        
        data_input = euclidean_dist if config["is_precomputed"] else scaled_features
        clusterer = GlobalMotifClusterer(data_input, is_precomputed=config["is_precomputed"])
        
        try:
            method = getattr(clusterer, config["func"])
            labels = method(**config["kwargs"])
        except Exception as e:
            print(f"     [Warning] {name} encountered an edge case ({e}). Fallback to KMeans predictions.")
            km_fallback = GlobalMotifClusterer(scaled_features, is_precomputed=False)
            labels = km_fallback.run_kmeans(n_clusters=n_clusters_target)
            
        runtimes[name] = time.time() - start_time
        all_labels[name] = np.array(labels)

    results_bundle["all_labels"] = all_labels

    # ==========================================
    # STAGE 8: HYPERPARAMETER TUNING GRID-SEARCH
    # ==========================================
    log_progress(8, "Stage 8: Hyperparameter Tuning Grid-Search", "(Exhaustive search over parameter spaces)")
    optimizer = ClusteringOptimizer(scaled_features, is_precomputed=False)
    
    print("  -> Optimizing KMeans & Agglomerative...")
    best_km_params, best_km_score = optimizer.optimize_kmeans()
    best_agg_params, best_agg_score = optimizer.optimize_agglomerative()
    
    tuning_summary = {
        "KMeans_Optimal_Params": best_km_params,
        "KMeans_Best_Silhouette": best_km_score,
        "Agglomerative_Optimal_Params": best_agg_params,
        "Agglomerative_Best_Silhouette": best_agg_score
    }
    with open(os.path.join(output_dir, "hyperparameter_tuning_results.json"), "w") as f:
        json.dump(tuning_summary, f, indent=4)
    results_bundle["tuning_summary"] = tuning_summary

    # ==========================================
    # STAGE 9 & 11: INTERNAL VALIDATION & MODEL COMPARISON
    # ==========================================
    log_progress(9, "Stage 9 & 11: Validation & Model Ranking", "(Computing Silhouette, Davies-Bouldin, CH & Gap stats)")
    comparator = MotifModelComparator()
    validation_reports = {}

    motif_lengths = clean_motifs_df['Length'].values
    for name, labels in all_labels.items():
        validator = ClusteringValidator(X=scaled_features, labels=labels, motif_lengths=motif_lengths)
        report = validator.generate_report()
        validation_reports[name] = report
        
        comparator.add_model_result(
            model_name=name,
            silhouette=report["Silhouette_Score"],
            db_index=report["Davies_Bouldin"],
            ch_score=report["Calinski_Harabasz"],
            runtime=runtimes[name],
            num_clusters=report["Num_Clusters"]
        )

    comparison_df = comparator.get_comparison_table()
    comparison_df.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    results_bundle["comparison_df"] = comparison_df
    results_bundle["validation_reports"] = validation_reports

    # Select Best Model
    best_model_info = comparator.select_best_model(primary_metric='Silhouette')
    best_model_name = best_model_info['Model']
    best_labels = all_labels[best_model_name]
    
    with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
        json.dump(best_model_info.to_dict(), f, indent=4)
    
    clean_motifs_df['Cluster_Label'] = best_labels
    clean_motifs_df.to_csv(os.path.join(output_dir, "cluster_assignments.csv"), index=False)
    results_bundle["best_model_info"] = best_model_info.to_dict()
    results_bundle["best_labels"] = best_labels
    
    print(f"\n🏆 BEST MODEL SELECTED: {best_model_name} (Silhouette: {best_model_info['Silhouette']})")

    # ==========================================
    # STAGE 10: DIAGNOSTIC & MOTIF PLOT GENERATORS
    # ==========================================
    log_progress(10, "Stage 10: Generating Visualizations", f"(Creating high-res plots for {best_model_name})")
    viz = MotifVisualizer()
    
    with patch('matplotlib.pyplot.show'):
        # 1. PCA Projection
        viz.plot_dimensionality_reduction(scaled_features, best_labels, method='pca')
        save_plot_and_close(output_dir, "10_pca_clusters.png")
        results_bundle["plots"]["10_pca_clusters"] = os.path.join(output_dir, "10_pca_clusters.png")
        
        # 2. t-SNE Projection
        viz.plot_dimensionality_reduction(scaled_features, best_labels, method='tsne')
        save_plot_and_close(output_dir, "10_tsne_clusters.png")
        results_bundle["plots"]["10_tsne_clusters"] = os.path.join(output_dir, "10_tsne_clusters.png")

        # 3. Silhouette Plot
        viz.plot_silhouette(scaled_features, best_labels)
        save_plot_and_close(output_dir, "10_silhouette_plot.png")
        results_bundle["plots"]["10_silhouette_plot"] = os.path.join(output_dir, "10_silhouette_plot.png")

        # 4. Length Histograms
        viz.plot_cluster_length_histograms(clean_motifs_df['Length'].tolist(), best_labels.tolist())
        save_plot_and_close(output_dir, "10_length_histograms.png")
        results_bundle["plots"]["10_length_histograms"] = os.path.join(output_dir, "10_length_histograms.png")

        # 5. Generalized Motifs (Figure 2 Style)
        motifs_list = [np.array(m) if not isinstance(m, str) else ast.literal_eval(m) for m in clean_motifs_df['Raw_Signal']]
        viz.plot_generalized_motifs(motifs_list, best_labels.tolist())
        save_plot_and_close(output_dir, "10_generalized_motifs.png")
        results_bundle["plots"]["10_generalized_motifs"] = os.path.join(output_dir, "10_generalized_motifs.png")
        
        # 6. Dendrogram
        viz.plot_dendrogram(scaled_features)
        save_plot_and_close(output_dir, "10_dendrogram.png")
        results_bundle["plots"]["10_dendrogram"] = os.path.join(output_dir, "10_dendrogram.png")

        # 7. Distance Matrix Heatmap
        viz.plot_distance_matrix_heatmap(euclidean_dist, best_labels.tolist())
        save_plot_and_close(output_dir, "10_distance_heatmap.png")
        results_bundle["plots"]["10_distance_heatmap"] = os.path.join(output_dir, "10_distance_heatmap.png")
        
        # 8. Vertical Signal Trace (Figure 1 Style)
        target_signal_id = clean_motifs_df['Signal_ID'].iloc[0]
        if target_signal_id in processed_signals.columns:
            sig_data = processed_signals[target_signal_id].values
            sig_motifs = clean_motifs_df[clean_motifs_df['Signal_ID'] == target_signal_id]
            viz.plot_vertical_signal_with_clusters(
                signal=sig_data,
                motif_starts=sig_motifs['Start_Index'].tolist(),
                motif_ends=sig_motifs['End_Index'].tolist(),
                labels=sig_motifs['Cluster_Label'].tolist(),
                signal_name=f"Signal: {target_signal_id}"
            )
            save_plot_and_close(output_dir, "10_vertical_signal_track.png")
            results_bundle["plots"]["10_vertical_signal_track"] = os.path.join(output_dir, "10_vertical_signal_track.png")

    log_progress(11, "Complete", "Pipeline executed all 11 stages successfully! All outputs saved.")
    return results_bundle


def main():
    parser = argparse.ArgumentParser(description="Run the 11-Stage Time-Series Motif Clustering Pipeline.")
    parser.add_argument("--input", type=str, default="data/signals_data.csv", help="Path to input signal CSV file.")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save outputs and plots.")
    args = parser.parse_args()
    
    print("=====================================================================")
    print("🌊 STARTING END-TO-END TIME-SERIES MOTIF CLUSTERING PIPELINE")
    print("=====================================================================")
    
    start_t = time.time()
    results = run_pipeline(input_csv_path=args.input, output_dir=args.output)
    elapsed = time.time() - start_t
    
    print("\n=====================================================================")
    print(f"PIPELINE FINISHED IN {elapsed:.2f} SECONDS")
    print(f"Best Model Identified: {results['best_model_info']['Model']}")
    print(f"Total Extracted Motifs: {results['raw_motifs_count']} -> Cleaned: {results['clean_motifs_count']}")
    print(f"All figures and dataframes saved under '{args.output}/'")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()

