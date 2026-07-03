# Hybrid Visual-Temporal Analysis for Lung Cancer Decision Support

A Multimodal Clinical Decision Support System that merges computer vision and longitudinal clinical analysis to assist in lung cancer evaluation and diagnostic decision-making. 

---

## Core Architecture
This system utilizes a hybrid visual-temporal approach to fuse medical imaging features with structured patient data:
* **Visual Classification**: Features custom deep learning architectures (`PulmoNetResNet`) built on top of a ResNet-18 backbone to process visual data from longitudinal CT-scan imaging.
* **Temporal & Clinical Modeling**: Integrates XGBoost and ARIMA models to process clinical metrics, text data, and historical patient histories.
* **Dimensionality Reduction**: Employs Principal Component Analysis (PCA) for efficient feature fusion across data modalities.

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
