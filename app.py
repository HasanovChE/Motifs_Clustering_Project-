import os
import ast
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the orchestrator function from main.py
from main import run_pipeline

# =========================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM CSS DESIGN SYSTEM
# =========================================================
st.set_page_config(
    page_title="Motif Clustering Studio | End-to-End Pipeline",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic and Premium Typography CSS
st.markdown("""
<style>
    /* Google Fonts & Base Theme */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4f46e5 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.5);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 800px;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.4rem;
        border-radius: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.15);
        border-color: rgba(79, 70, 229, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
        margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
    }
    
    /* Best Model Highlight Box */
    .best-model-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 95, 70, 0.15) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }
    
    /* Stage Pill Badges */
    .stage-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 0.5rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER & HERO BANNER
# =========================================================
st.markdown("""
<div class="hero-banner">
    <div class="stage-badge">🚀 Automated 11-Stage Pipeline Engine</div>
    <div class="hero-title">🌊 Time-Series Motif Clustering Studio</div>
    <div class="hero-subtitle">
        Upload multi-channel signal datasets or use default well/sensor logs. Automated execution across data cleaning, 
        multi-method motif extraction, wavelet/FFT feature engineering, similarity matrices, deep embedded clustering, and diagnostic evaluations.
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State for preserving execution results
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "current_csv_path" not in st.session_state:
    st.session_state["current_csv_path"] = "data/signals_data.csv"

# =========================================================
# SIDEBAR CONTROLS & DATA INGESTION
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/waveform.png", width=64)
    st.title("⚙️ Pipeline Controls")
    st.markdown("---")
    
    st.subheader("📂 1. Data Source Selection")
    data_source_option = st.radio(
        "Choose Input CSV Method:",
        options=["Use Default Dataset (`signals_data.csv`)", "Upload Custom Time-Series CSV"],
        index=0
    )
    
    selected_csv_path = "data/signals_data.csv"
    
    if data_source_option == "Upload Custom Time-Series CSV":
        uploaded_file = st.file_uploader("Upload Signal CSV (Columns = Signals)", type=["csv"])
        if uploaded_file is not None:
            os.makedirs("data", exist_ok=True)
            custom_path = os.path.join("data", "uploaded_signals_data.csv")
            with open(custom_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_csv_path = custom_path
            st.success(f"✅ Uploaded: `{uploaded_file.name}`")
        else:
            st.info("💡 Please upload a CSV file or switch to default dataset.")
            
    st.session_state["current_csv_path"] = selected_csv_path
    
    # Preview current dataset info
    if os.path.exists(selected_csv_path):
        try:
            df_preview = pd.read_csv(selected_csv_path, nrows=5)
            st.caption(f"**Loaded File:** `{os.path.basename(selected_csv_path)}`")
            st.caption(f"**Grid Shape:** {pd.read_csv(selected_csv_path).shape[0]} rows × {df_preview.shape[1]} columns")
        except Exception:
            pass
            
    st.markdown("---")
    st.subheader("⚡ 2. Automated Execution")
    
    run_btn = st.button("🚀 Run Full 11-Stage Pipeline", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🧩 Pipeline Architecture")
    st.markdown("""
    - **Stage 1:** EDA & Profiling
    - **Stage 2:** Noise & Baseline Removal
    - **Stage 3:** Multi-Method Motif Extraction
    - **Stage 4:** Anomaly & Outlier Cleaning
    - **Stage 5:** 25+ Feature Engineering
    - **Stage 6:** Distance/Similarity Matrices
    - **Stage 7:** Global Clustering Suite
    - **Stage 8:** Grid-Search Hyper Tuning
    - **Stage 9:** Internal Validation Score
    - **Stage 10:** High-Res Diagnostic Plots
    - **Stage 11:** Best Model Selection
    """)

# =========================================================
# PIPELINE EXECUTION IN BROWSER
# =========================================================
if run_btn:
    if not os.path.exists(st.session_state["current_csv_path"]):
        st.error(f"❌ Input file not found at `{st.session_state['current_csv_path']}`. Please check selection.")
    else:
        st.session_state["pipeline_results"] = None
        status_container = st.status("⚡ Running 11-Stage Motif Clustering Pipeline...", expanded=True)
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Callback to update UI during live pipeline processing
        def ui_progress_callback(stage_idx, stage_name, message):
            status_container.update(label=f"⏳ Stage [{stage_idx}/11]: {stage_name}", state="running")
            status_text.markdown(f"**Current Operation:** {message}")
            progress_bar.progress(min(1.0, stage_idx / 11.0))
            
        try:
            start_t = time.time()
            results = run_pipeline(
                input_csv_path=st.session_state["current_csv_path"],
                output_dir="outputs",
                progress_callback=ui_progress_callback
            )
            elapsed_time = time.time() - start_t
            results["elapsed_time"] = elapsed_time
            
            status_container.update(label=f"✅ Pipeline Completed Successfully in {elapsed_time:.2f}s!", state="complete")
            progress_bar.progress(1.0)
            status_text.markdown(f"**All 11 stages processed.** Best Model: `{results['best_model_info']['Model']}`")
            
            st.session_state["pipeline_results"] = results
            st.toast("🎉 Pipeline execution complete! Explore tabs below.")
            
        except Exception as e:
            status_container.update(label="❌ Pipeline Failed during Execution", state="error")
            st.error(f"Execution Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# =========================================================
# RESULTS DASHBOARD TABS
# =========================================================
results = st.session_state["pipeline_results"]

if results is None:
    # Landing State when no pipeline has been run yet
    st.info("👈 **Welcome!** Please select a dataset in the sidebar and click **'🚀 Run Full 11-Stage Pipeline'** to begin automated browser execution and analysis.")
    
    if os.path.exists(st.session_state["current_csv_path"]):
        st.subheader("📋 Quick Preview of Selected Dataset")
        df_init = pd.read_csv(st.session_state["current_csv_path"])
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Signal Channels</div><div class="metric-value">{df_init.shape[1]}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Time Steps / Rows</div><div class="metric-value">{df_init.shape[0]}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Data Points</div><div class="metric-value">{df_init.size:,}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Missing / Null Values</div><div class="metric-value">{df_init.isnull().sum().sum()}</div></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        st.dataframe(df_init.head(20), use_container_width=True)
else:
    # Render the Full 7-Tab Interactive Dashboard
    st.markdown("### 📊 Interactive Pipeline Results Dashboard")
    
    tabs = st.tabs([
        "🌐 Overview & EDA (Stg 1-2)",
        "🔍 Extracted Motifs (Stg 3-4)",
        "⚙️ Features & Similarity (Stg 5-6)",
        "🤖 Clustering & Tuning (Stg 7-8)",
        "🏆 Model Comparison (Stg 9 & 11)",
        "📈 Diagnostic Plots (Stg 10)",
        "💾 Export Deliverables"
    ])
    
    # -----------------------------------------------------
    # TAB 1: OVERVIEW & EDA (Stage 1 & 2)
    # -----------------------------------------------------
    with tabs[0]:
        st.subheader("Stage 1: Exploratory Data Analysis & Stage 2: Signal Preprocessing")
        
        raw_df = results["raw_df"]
        processed_df = results["processed_signals"]
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Raw Channels</div><div class="metric-value">{raw_df.shape[1]}</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Time Steps</div><div class="metric-value">{raw_df.shape[0]}</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Null Values</div><div class="metric-value">{raw_df.isnull().sum().sum()}</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Processing Time</div><div class="metric-value">{results.get('elapsed_time', 0):.1f}s</div></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### 🔬 Interactive Signal Track & Preprocessing Viewer")
        
        selected_col = st.selectbox("Select Signal Track to Inspect:", options=raw_df.columns, index=0)
        
        col_view1, col_view2 = st.columns([1, 1])
        with col_view1:
            st.markdown(f"**Original Raw Signal (`{selected_col}`)**")
            st.line_chart(raw_df[selected_col], height=260)
        with col_view2:
            st.markdown(f"**Preprocessed Signal (`{selected_col}` - Savitzky-Golay + Polynomial Baseline)**")
            if selected_col in processed_df.columns:
                st.line_chart(processed_df[selected_col], height=260)
            else:
                st.info("Preprocessed track not available for this column.")
                
        st.markdown("---")
        st.markdown("#### 🖼️ Preprocessing Filter Comparison Tracks (Depth-Log Layout)")
        if "02_preprocessing_comparison" in results["plots"] and os.path.exists(results["plots"]["02_preprocessing_comparison"]):
            st.image(results["plots"]["02_preprocessing_comparison"], caption="Stage 2: Comparison of Noise Removal, Baseline Subtraction, and Normalization Techniques", use_container_width=True)

    # -----------------------------------------------------
    # TAB 2: EXTRACTED & CLEANED MOTIFS (Stage 3 & 4)
    # -----------------------------------------------------
    with tabs[1]:
        st.subheader("Stage 3: Multi-Method Motif Extraction & Stage 4: Outlier Cleaning")
        
        clean_motifs_df = results["clean_motifs_df"]
        raw_count = results["raw_motifs_count"]
        clean_count = results["clean_motifs_count"]
        retention = (clean_count / raw_count * 100) if raw_count > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Raw Candidate Motifs</div><div class="metric-value">{raw_count}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Cleaned Retained Motifs</div><div class="metric-value">{clean_count}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Retention Rate</div><div class="metric-value">{retention:.1f}%</div></div>""", unsafe_allow_html=True)
        with c4:
            avg_len = clean_motifs_df['Length'].mean() if not clean_motifs_df.empty else 0
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Mean Motif Length</div><div class="metric-value">{avg_len:.1f}</div></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        
        if "extraction_method" in clean_motifs_df.columns:
            st.markdown("#### 📈 Motifs by Extraction Strategy")
            method_counts = clean_motifs_df['extraction_method'].value_counts()
            st.bar_chart(method_counts)
            
        st.markdown("#### 📋 Cleaned Candidate Motifs Table")
        st.dataframe(clean_motifs_df.drop(columns=['Raw_Signal'], errors='ignore'), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🔎 Interactive Single Motif Shape Inspector")
        if not clean_motifs_df.empty:
            motif_ids = clean_motifs_df['Motif_ID'].tolist()
            selected_motif_id = st.selectbox("Select Motif ID to view waveform:", options=motif_ids)
            
            row = clean_motifs_df[clean_motifs_df['Motif_ID'] == selected_motif_id].iloc[0]
            raw_sig_data = row['Raw_Signal']
            if isinstance(raw_sig_data, str):
                raw_sig_data = ast.literal_eval(raw_sig_data)
                
            fig_motif, ax_motif = plt.subplots(figsize=(10, 3))
            ax_motif.plot(raw_sig_data, color='#6366f1', linewidth=2)
            ax_motif.set_title(f"Waveform for {selected_motif_id} (Channel: {row.get('Signal_ID', 'N/A')}, Length: {len(raw_sig_data)})", fontweight='bold')
            ax_motif.set_xlabel("Time Step Index")
            ax_motif.set_ylabel("Amplitude")
            ax_motif.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig_motif)

    # -----------------------------------------------------
    # TAB 3: FEATURE ENGINEERING & SIMILARITY (Stage 5 & 6)
    # -----------------------------------------------------
    with tabs[2]:
        st.subheader("Stage 5: Feature Engineering Suite & Stage 6: Similarity Matrices")
        
        features_df = results["features_df"]
        st.markdown(f"**Total Features Extracted per Motif:** `{features_df.shape[1] - 2}` (Statistical, Shape, Frequency/FFT, Wavelet/DWT, Autocorrelation)")
        
        st.dataframe(features_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🗺️ Precomputed Euclidean Distance Matrix Heatmap")
        if "euclidean_dist" in results:
            dist_mat = results["euclidean_dist"]
            fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
            cax = ax_hm.imshow(dist_mat, cmap='viridis_r', aspect='auto')
            fig_hm.colorbar(cax, label="Euclidean Distance")
            ax_hm.set_title("Pairwise Distance Matrix across All Cleaned Motifs")
            ax_hm.set_xlabel("Motif Index")
            ax_hm.set_ylabel("Motif Index")
            st.pyplot(fig_hm)

    # -----------------------------------------------------
    # TAB 4: CLUSTERING & HYPER TUNING (Stage 7 & 8)
    # -----------------------------------------------------
    with tabs[3]:
        st.subheader("Stage 7: Global Clustering Models & Stage 8: Grid-Search Hyperparameter Optimization")
        
        st.markdown("#### 🤖 Execution Runtimes across Multi-Algorithm Suite")
        comp_df = results["comparison_df"]
        st.dataframe(comp_df[['Model', 'Number of Clusters', 'Runtime (s)', 'Silhouette', 'DB Index', 'CH Score']], use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### ⚙️ Stage 8: Grid-Search Hyperparameter Tuning Summary")
        if "tuning_summary" in results and results["tuning_summary"]:
            t_sum = results["tuning_summary"]
            col_tune1, col_tune2 = st.columns(2)
            with col_tune1:
                st.markdown("##### 📌 KMeans Optimal Parameters")
                st.json({
                    "Optimal_Hyperparameters": t_sum.get("KMeans_Optimal_Params"),
                    "Best_Silhouette_Score": t_sum.get("KMeans_Best_Silhouette")
                })
            with col_tune2:
                st.markdown("##### 📌 Agglomerative Clustering Optimal Parameters")
                st.json({
                    "Optimal_Hyperparameters": t_sum.get("Agglomerative_Optimal_Params"),
                    "Best_Silhouette_Score": t_sum.get("Agglomerative_Best_Silhouette")
                })

    # -----------------------------------------------------
    # TAB 5: MODEL COMPARISON & BEST MODEL (Stage 9 & 11)
    # -----------------------------------------------------
    with tabs[4]:
        st.subheader("Stage 9: Internal Validation Metrics & Stage 11: Automated Model Selection")
        
        best_info = results["best_model_info"]
        
        # Best Model Highlight Box
        st.markdown(f"""
        <div class="best-model-box">
            <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #10b981;">🏆 Automated Stage 11 Selection Winner</div>
            <div style="font-size: 2.2rem; font-weight: 700; margin: 0.4rem 0; color: #ffffff;">{best_info['Model']}</div>
            <div style="display: flex; gap: 2.5rem; margin-top: 1rem; flex-wrap: wrap;">
                <div><span style="opacity: 0.8;">Silhouette Score:</span> <strong style="font-size: 1.3rem; color: #34d399;">{best_info['Silhouette']:.4f}</strong></div>
                <div><span style="opacity: 0.8;">Davies-Bouldin Index:</span> <strong style="font-size: 1.3rem; color: #6ee7b7;">{best_info['DB Index']:.4f}</strong></div>
                <div><span style="opacity: 0.8;">Calinski-Harabasz:</span> <strong style="font-size: 1.3rem; color: #a7f3d0;">{best_info['CH Score']:.1f}</strong></div>
                <div><span style="opacity: 0.8;">Number of Clusters:</span> <strong style="font-size: 1.3rem; color: #ffffff;">{best_info['Number of Clusters']}</strong></div>
                <div><span style="opacity: 0.8;">Compute Time:</span> <strong style="font-size: 1.3rem; color: #ffffff;">{best_info['Runtime (s)']:.3f}s</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🏁 Complete Evaluation Leaderboard")
        st.dataframe(comp_df.sort_values("Silhouette", ascending=False), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📈 Cluster Size & Length Distribution (Best Model)")
        if best_info['Model'] in results.get("validation_reports", {}):
            val_rep = results["validation_reports"][best_info['Model']]
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("##### Cluster Size Counts & Percentages")
                clean_df = results["clean_motifs_df"]
                if "Cluster_Label" in clean_df.columns:
                    size_counts = clean_df["Cluster_Label"].value_counts().reset_index()
                    size_counts.columns = ["Cluster ID", "Motif Count"]
                    size_counts["Percentage (%)"] = (size_counts["Motif Count"] / len(clean_df) * 100).round(1)
                    st.dataframe(size_counts, use_container_width=True)
            with col_v2:
                st.markdown("##### Motif Length Statistics across Clusters")
                if "Cluster_Label" in clean_df.columns:
                    len_stats = clean_df.groupby("Cluster_Label")["Length"].agg(["mean", "std", "min", "max"]).round(1).reset_index()
                    len_stats.columns = ["Cluster ID", "Mean Len", "Std Len", "Min Len", "Max Len"]
                    st.dataframe(len_stats, use_container_width=True)

    # -----------------------------------------------------
    # TAB 6: DIAGNOSTIC PLOTS (Stage 10)
    # -----------------------------------------------------
    with tabs[5]:
        st.subheader("Stage 10: High-Resolution Diagnostic & Cluster Visualizations")
        
        plots = results.get("plots", {})
        
        if "10_vertical_signal_track" in plots and os.path.exists(plots["10_vertical_signal_track"]):
            st.markdown("#### 1. Vertical Signal Depth-Log Track with Cluster Overlays (Figure 1 Style)")
            st.image(plots["10_vertical_signal_track"], caption="Vertical continuous signal log showing extracted motif zones shaded by cluster assignment", use_container_width=False, width=450)
            st.markdown("---")
            
        if "10_generalized_motifs" in plots and os.path.exists(plots["10_generalized_motifs"]):
            st.markdown("#### 2. Generalized Cluster Shapes & Candidate Overlays (Figure 2 Style)")
            st.image(plots["10_generalized_motifs"], caption="Individual resampled motifs (grey) overlaid with the cluster centroid generalized shape (bold color)", use_container_width=True)
            st.markdown("---")
            
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if "10_pca_clusters" in plots and os.path.exists(plots["10_pca_clusters"]):
                st.markdown("#### 3. PCA 2D Manifold Projection")
                st.image(plots["10_pca_clusters"], use_container_width=True)
        with col_p2:
            if "10_tsne_clusters" in plots and os.path.exists(plots["10_tsne_clusters"]):
                st.markdown("#### 4. t-SNE 2D Manifold Projection")
                st.image(plots["10_tsne_clusters"], use_container_width=True)
                
        st.markdown("---")
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            if "10_silhouette_plot" in plots and os.path.exists(plots["10_silhouette_plot"]):
                st.markdown("#### 5. Silhouette Coefficient Profile")
                st.image(plots["10_silhouette_plot"], use_container_width=True)
        with col_p4:
            if "10_length_histograms" in plots and os.path.exists(plots["10_length_histograms"]):
                st.markdown("#### 6. Length Distribution Histograms")
                st.image(plots["10_length_histograms"], use_container_width=True)
                
        st.markdown("---")
        col_p5, col_p6 = st.columns(2)
        with col_p5:
            if "10_dendrogram" in plots and os.path.exists(plots["10_dendrogram"]):
                st.markdown("#### 7. Hierarchical Dendrogram")
                st.image(plots["10_dendrogram"], use_container_width=True)
        with col_p6:
            if "10_distance_heatmap" in plots and os.path.exists(plots["10_distance_heatmap"]):
                st.markdown("#### 8. Distance Matrix Cluster Blocks")
                st.image(plots["10_distance_heatmap"], use_container_width=True)

    # -----------------------------------------------------
    # TAB 7: EXPORT DELIVERABLES
    # -----------------------------------------------------
    with tabs[6]:
        st.subheader("💾 Export Pipeline Output Files & Deliverables")
        st.markdown("Download generated CSV reports, feature matrices, model comparison tables, and cluster assignments.")
        
        output_dir = results["output_dir"]
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            if os.path.exists(os.path.join(output_dir, "cleaned_motifs.csv")):
                with open(os.path.join(output_dir, "cleaned_motifs.csv"), "rb") as f:
                    st.download_button(
                        label="📥 Download Cleaned Motifs (`cleaned_motifs.csv`)",
                        data=f,
                        file_name="cleaned_motifs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            if os.path.exists(os.path.join(output_dir, "feature_matrix.csv")):
                with open(os.path.join(output_dir, "feature_matrix.csv"), "rb") as f:
                    st.download_button(
                        label="📥 Download Feature Matrix (`feature_matrix.csv`)",
                        data=f,
                        file_name="feature_matrix.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        with col_ex2:
            if os.path.exists(os.path.join(output_dir, "evaluation_metrics.csv")):
                with open(os.path.join(output_dir, "evaluation_metrics.csv"), "rb") as f:
                    st.download_button(
                        label="📥 Download Model Evaluation Metrics (`evaluation_metrics.csv`)",
                        data=f,
                        file_name="evaluation_metrics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            if os.path.exists(os.path.join(output_dir, "cluster_assignments.csv")):
                with open(os.path.join(output_dir, "cluster_assignments.csv"), "rb") as f:
                    st.download_button(
                        label="📥 Download Cluster Assignments (`cluster_assignments.csv`)",
                        data=f,
                        file_name="cluster_assignments.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        with col_ex3:
            if os.path.exists(os.path.join(output_dir, "best_model_info.json")):
                with open(os.path.join(output_dir, "best_model_info.json"), "rb") as f:
                    st.download_button(
                        label="📥 Download Best Model JSON (`best_model_info.json`)",
                        data=f,
                        file_name="best_model_info.json",
                        mime="application/json",
                        use_container_width=True
                    )
            if os.path.exists(os.path.join(output_dir, "hyperparameter_tuning_results.json")):
                with open(os.path.join(output_dir, "hyperparameter_tuning_results.json"), "rb") as f:
                    st.download_button(
                        label="📥 Download Hyper Tuning Results (`hyperparameter_tuning_results.json`)",
                        data=f,
                        file_name="hyperparameter_tuning_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
