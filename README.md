# Hybrid Visual-Temporal Analysis for Lung Cancer Decision Support

A Multimodal Clinical Decision Support System that fuses computer vision and longitudinal clinical analysis to assist in lung cancer evaluation, trained on the **National Lung Screening Trial (NLST)** dataset.

---

## The Web Application
The system features an interactive Flask-based web dashboard (`Application.py`) designed for clinical deployment:
* **Real-Time CT Scan Analysis**: Upload patient CT slices to generate immediate malignancy probability predictions via the integrated deep learning pipeline.
* **Multimodal Data Fusion**: Simultaneously processes structural clinical records, patient histories, and visual imaging data.
* **Clinical UI**: Interactive frontend components (`templates/index.html`) optimized for displaying diagnostic insights and feature extraction tracking.

---

## Core Architecture & Methodology

### Visual Branch (`PulmoNetResNet`)
A custom architecture built on a ResNet-18 backbone to extract 128-dimensional spatial feature embeddings using a **two-phase training protocol**:
* **Phase 1 (Pre-training)**: Trained as a binary classifier on the high-resolution **Chest CT-Scan Images** and **IQ-OTH/NCCD** Kaggle datasets to build a strong baseline vocabulary of pulmonary pathology. 
  * *Parameters*: Adam ($lr = 1 \times 10^{-3}$), Cross-Entropy Loss, 10 epochs. Saves peak iteration as `best_pulmonet_standalone.pth`.
* **Phase 2 (Fine-tuning)**: Transitioned to the **NLST cohort** to capture longitudinal visual shifts across screening timelines.

### Temporal & Clinical Branch
* **Data Merging**: Systematically maps 5 relational NLST metadata tables (`canc`, `ctab`, `ctabc`, `prsn`, `screen`) on shared `pid` and `study_yr` keys to yield 3 ordered sequential rows per patient for 395 unique subjects.
* **Imputation Strategy**: Missing values are clinically handled (nodule dimensions $\rightarrow$ `0`, staging $\rightarrow$ `0`, cancer-free days $\rightarrow$ max column survival, hardware metrics $\rightarrow$ column medians).
* **Downstream Modeling**: Leverages an engineered pipeline of **XGBoost**, **ARIMA**, and **PCA** to fuse features and execute diagnostic predictions.

---

## Project Structure
```text
├── articles/              # Domain research papers and documentation
├── templates/             # HTML files for the Flask application interface
├── Application.py         # Main web application backend and model routing
├── Preprocessing_CT.ipynb # CT image normalization and slice processing pipelines
├── data_preprocessed.ipynb# Clinical metadata feature engineering notebook
├── final_code(...).ipynb  # Training scripts integrating ResNet18 + ARIMA + XGBoost
└── *.csv / *.json         # Processed test targets and metadata configurations
