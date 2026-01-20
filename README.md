# SMS Spam Detection — NLP Assignment (Baseline, CNN, DistilBERT)

This repository contains my NLP assignment for **SMS spam detection** (binary text classification):  
**ham (0)** vs **spam (1)**.

I implement and compare three models under the same data split and evaluation protocol:

- **Baseline (Classical ML):** TF-IDF (word n-grams) + LinearSVC  
- **Deep Learning:** CNN for text classification  
- **Transformer:** DistilBERT fine-tuning for sequence classification  

The repo also includes **evaluation metrics**, **plots** (Confusion Matrix, ROC curve, learning curve), and **error analysis** (False Positives / False Negatives examples).

---

## Repository Structure
---

## Dataset

The code expects a cleaned CSV in `data/` with columns:
- `text` (string)
- `label` (int: 0=ham, 1=spam)

and a saved split file (e.g. `split_idx.npz`) with:
- `train_idx`, `val_idx`, `test_idx`

> If the dataset is not included in the repo, run `data.ipynb` to prepare it (or add the dataset manually into `data/` following the notebook instructions).

---

## Installation

Create an environment and install dependencies:


How to Run

Suggested order:
	1.	Data preparation
	•	Run: data.ipynb
	•	Output: cleaned dataset + split indices inside data/
	2.	Baseline model
	•	Run: baseline.ipynb
	•	Output: metrics/plots saved under metrics/ and plots/
	3.	CNN model
	•	Run: deep_cnn.ipynb
	•	Output: metrics/plots (+ optional saved model) under metrics/, plots/, error_analysis/
	4.	DistilBERT
	•	Run: transformer_distilbert.ipynb
	•	Output: metrics/plots and checkpoint info under metrics/, plots/, error_analysis/
	5.	Error analysis
	•	Run: error_analysis.ipynb
	•	Output: FP/FN CSV files and interpretability outputs under error_analysis/

  Evaluation

Because the dataset is imbalanced, evaluation focuses on:
	•	Accuracy
	•	Precision / Recall / F1-score for the spam class
	•	ROC-AUC
	•	Confusion Matrix and ROC curve plots

Metrics are saved in metrics/ and key plots in plots/.

Results

Produced artifacts (examples):
	•	Metrics (CSV): metrics/*_metrics_*.csv
	•	Plots (PNG): plots/*confusion_matrix*.png, plots/*roc_curve*.png, plots/*learning*.png
	•	Error analysis: error_analysis/*false_positives*.csv, error_analysis/*false_negatives*.csv
	•	Interpretability (optional): error_analysis/lime_*.html

  
