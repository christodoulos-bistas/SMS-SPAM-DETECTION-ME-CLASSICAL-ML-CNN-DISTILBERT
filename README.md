# Spam SMS Detection — NLP Assignment (Baseline, CNN, DistilBERT)

This project tackles **SMS spam detection** (binary text classification):  
**ham (0)** vs **spam (1)**.

I implement and compare three approaches under the same evaluation protocol:

- **Baseline:** TF-IDF (word n-grams) + LinearSVC  
- **CNN:** Text CNN (Embedding → Conv1D → Pooling → Dense)  
- **Transformer:** DistilBERT fine-tuning for sequence classification  

The repository also includes **visualizations** (Confusion Matrices, ROC Curves, CNN learning curve), **error analysis** (FP/FN examples), and **interpretability** (LIME for the baseline).

---

## Contents

- `notebooks/`
  - `data.ipynb` — data preparation / splits
  - `baseline.ipynb` — TF-IDF + LinearSVC training & evaluation
  - `deep_cnn.ipynb` — CNN training & evaluation
  - `transformer_distilbert.ipynb` — DistilBERT fine-tuning & evaluation
  - `error_analysis.ipynb` — FP/FN analysis + interpretability outputs
- `data/` — cleaned dataset + saved split indices (or download instructions)
- `results/` — metrics, plots, and saved artifacts (CSV/PNG/HTML)
- `report/` — final report document
- `slides/` — presentation deck

> If your repo uses a slightly different structure, update the section titles accordingly.

---

## Dataset

Expected format (after cleaning):

- `data/sms_clean.csv` with columns: `text`, `label` where `label ∈ {0,1}`
- `data/split_idx.npz` containing `train_idx`, `val_idx`, `test_idx`

If the dataset is not stored in the repository, run `notebooks/data.ipynb` to generate the cleaned file and splits (or follow the dataset download steps described there).

---

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
