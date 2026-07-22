# 🌊 Time-Series Motif Clustering Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Signal Processing pipeline designed to extract, clean, feature-engineer, cluster, and visualize recurring time-series pattern candidates (**motifs**) across multi-channel signal datasets (e.g., well-log data, seismic traces, or sensor feeds).

Includes a full **CLI pipeline** and an interactive **Streamlit Web Application** for automated deployment.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
  - [1. Running via Command Line (CLI)](#1-running-via-command-line-cli)
  - [2. Launching the Web App (Streamlit)](#2-launching-the-web-app-streamlit)
- [Pipeline Stages](#-pipeline-stages)
- [Outputs & Deliverables](#-outputs--deliverables)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🔍 Overview

Time-series signals often contain recurring structural shapes or patterns called **motifs**. Identifying and grouping these motifs across hundreds of signal columns manually is time-consuming and error-prone. 

This repository provides an automated, 10-stage end-to-end framework that takes raw signal CSV data, isolates candidate motif zones based on flexible thresholding strategies, computes similarity metrics (including **DTW** and **Shape-Based Distance**), performs global clustering using classical and deep learning models, and evaluates the optimal results.

---

## ✨ Key Features

- **Multi-Method Motif Extraction**: Fixed amplitude thresholding, adaptive statistical thresholds ($\mu + k \cdot \sigma$), peak detection, and derivative-based slope analysis.
- **Robust Cleaning**: Automatic filtering of noise, constant/flat signals, duplicates, and statistical outliers.
- **Comprehensive Feature Extraction**: Over 25+ statistical, shape, frequency (FFT), wavelet (DWT), and autocorrelation (ACF/PACF) features.
- **Dynamic Similarity Measures**: Euclidean, Manhattan, Cosine, Correlation, Soft-DTW, Shape-Based Distance (SBD), and custom Length-Penalized Hybrid Distance.
- **Extensive Clustering Suite**: KMeans, K-Medoids, Agglomerative, DBSCAN, HDBSCAN, Spectral Clustering, SOM, Autoencoder + KMeans, and Deep Embedded Clustering (DEC).
- **Automated Model Selection**: Grid-search hyperparameter optimization evaluated against Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and Dunn Index.
- **Rich Visualizations**: Vertical depth-log tracks, generalized cluster overlays, PCA/t-SNE/UMAP projections, and length distribution histograms.

---

## 📂 Simply Project Architecture


├── signals_data.csv     # Input signal dataset (500 columns x N rows)
├── eda.py               # Stage 0: Exploratory Data Analysis & Profiling
├── preprocessing.py     # Stage 1: Noise/Baseline Removal & Normalization
├── extraction.py        # Stage 2: Column-wise Motif Extraction
├── cleaning.py          # Stage 3: Anomaly & Outlier Motif Filtering
├── features.py          # Stage 4: Statistical, Frequency & Wavelet Features
├── similarity.py        # Stage 5: Distance Matrix Computation (DTW/SBD)
├── clustering.py        # Stage 6: Global Clustering Implementations
├── optimization.py      # Stage 7: Hyperparameter Tuning Grid-Search
├── validation.py        # Stage 8: Internal Metrics & Gap Statistic
├── visualization.py     # Stage 9: Diagnostic & Motif Plot Generators
├── model_comparison.py  # Stage 10: Model Ranking & Best Model Selection
├── main.py              # Automated CLI Pipeline Orchestrator
├── app.py               # Streamlit Web Application Interface
├── requirements.txt     # Python Dependency Manifest
└── outputs/             # Auto-generated directory for saved outputs