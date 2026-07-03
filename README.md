# Hybrid Visual-Temporal Analysis for Lung Cancer Decision Support using the NLST Dataset

A Multimodal Clinical Decision Support System that merges computer vision and longitudinal clinical analysis to assist in lung cancer evaluation and diagnostic decision-making, trained and validated using the National Lung Screening Trial (NLST) dataset.

---

## Core Architecture & Training Methodology

This system utilizes a hybrid visual-temporal approach to fuse medical imaging features with structured patient data. 

### Visual Classification (Two-Phase Strategy)
The spatial feature extraction branch utilizes our custom **`PulmoNetResNet`** architecture built on top of a ResNet-18 backbone. It extracts optimized 128-dimensional spatial feature embeddings using a sophisticated two-phase training protocol:

* **Phase 1: Pre-training on Public Datasets**
  To build a generalizable vocabulary of pulmonary pathology and establish robust spatial weights, the model was pre-trained on two public Kaggle datasets:
  * **Chest CT-Scan Images Dataset**: Provides high-resolution slices across four histopathological classes (Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, and Normal). The model is explicitly configured as a **binary classifier (Malignant vs. Benign/Normal)** to align with downstream tasks while leveraging the rich morphological diversity.
  * **IQ-OTH/NCCD Dataset**: Contributes 1,190 clinically validated images from an active oncology setting, exposing the network to complex features like fine spiculation at nodule margins and subtle density gradations mimicking benign granulomas.
  * *Training Configurations*: 80/20 train/validation split, Adam optimizer ($lr = 1 \times 10^{-3}$), Cross-Entropy Loss, executed over 10 epochs. A local checkpointing mechanism automatically saves the peak-performing iteration as `best_pulmonet_standalone.pth`.

* **Phase 2: Longitudinal Fine-Tuning on NLST**
  With the spatial extractor established, the pipeline transitions to the longitudinal **National Lung Screening Trial (NLST)** cohort tracking visual shifts across screening timelines.

### Temporal, Metadata & Clinical Modeling
* **Data Integration Pipeline**: Patient timelines are systematically reconstructed by merging 5 foundational NLST relational tables (`canc`, `ctab`, `ctabc`, `prsn`, and `screen`) mapped on shared `pid` and `study_yr` primary keys. This generates a clean, sequential tracking profile (3 ordered rows per patient) for 395 unique subjects.
* **Clinical Imputation Strategy**: 
  * Missing nodule dimensions $\rightarrow$ Imputed with `0` (indicates no nodule detected).
  * Missing cancer stagings $\rightarrow$ Imputed with `0` (indicates no current malignancy).
  * Missing cancer-free days $\rightarrow$ Imputed with column maximum (complete survival status).
  * Missing continuous technical/hardware parameters $\rightarrow$ Imputed with their respective column medians.
* **Downstream Fusion**: Integrates an engineered combination of **XGBoost** and **ARIMA** models paired with **Principal Component Analysis (PCA)** to execute high-fidelity multimodal clinical prediction.

---

## Project Structure
```text
├── articles/              # Relevant domain research papers and documentation
├── templates/             # HTML templates for the web application UI
├── Application.py         # Main web application backend framework
├── Preprocessing_CT.ipynb # CT-scan image normalization and slicing pipelines
├── data_preprocessed.ipynb# Structured clinical data engineering notebook
├── final_code(...).ipynb  # Core training scripts integrating ResNet18 + ARIMA + XGBoost
└── *.csv / *.json         # Processed test sets and anonymized clinical targets
