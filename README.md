# 202418050_DL_Project

A Python-based deep learning pipeline for temporal sentiment analysis on Amazon product reviews. This project provides data ingestion, baseline models (Random Forest and LSTM), and a custom temporal sentiment model to capture evolving sentiment trends over time.

Features
Data Collection & Preprocessing
Downloads Amazon review data, cleans text, tokenizes, and constructs time-aware datasets.

Baseline Models
– Random Forest classifier for sentiment prediction.
– LSTM network with embedding, dropout, and adjustable hidden layers.

Temporal Sentiment Model
Integrates temporal embeddings and sequential attention to model how sentiment changes over time.

Installation
Clone the repository:

bash
git clone https://github.com/Shashwat141/202418050_DL_Project.git
cd 202418050_DL_Project
Create a virtual environment and install dependencies:

bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_fixed.txt
Verify setup:

bash
python gpucheck.py
python verify_setup.py
Usage
Download data

bash
python download_amazon_data.py --output data/raw_reviews.csv
Prepare dataset

bash
python dataset_final.py --input data/raw_reviews.csv --output data/processed.pkl
Train baselines

bash
python run_baselines_fixed.py --data data/processed.pkl
Train temporal model

bash
python train_final_fixed.py --data data/processed.pkl --model temporal_sentiment_model_final.py
Generate report

bash
python generate_report.py --predictions results/predictions.csv --output report.pdf
File Overview
download_amazon_data.py – Fetches and stores raw review data.

dataset_final.py – Cleans, tokenizes, and serializes time-stamped datasets.

baseline_rf_final.py / baseline_lstm_fixed.py – Implements Random Forest and LSTM baselines.

temporal_sentiment_model_final.py – Defines the custom temporal sentiment architecture.

train_final_fixed.py – Training loop for the temporal model, with checkpointing and evaluation.

generate_report.py – Creates performance report with metrics and visualizations.

verify_setup.py / gpucheck.py – Environment and GPU readiness checks.
