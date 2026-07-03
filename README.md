# Hybrid Visual-Temporal Analysis for Lung Cancer Decision Support

A multimodal clinical decision support system that fuses **computer vision** (CT imaging) and **longitudinal clinical analysis** to assist in lung cancer evaluation. Trained on the **National Lung Screening Trial (NLST)** cohort.

## The Web Application

An interactive Flask dashboard (`Application.py`):
- **Real-Time CT Analysis** — upload a zipped DICOM series → malignancy probability via the full pipeline
- **Multimodal Fusion** — combines imaging features with clinical variables (age, gender, smoking history)
- **AI Clinical Narrative** — Gemini API generates a short clinical note from the results
- **Clinical UI** — `templates/index.html` displays diagnosis, staging, and prognosis

## Pipeline

```
DICOM ZIP → HU normalization → PulmoNetResNet (ResNet-18) → 128-D embedding
   → temporal vector (ARIMA-style forecast + velocity) → PCA (10 components)
   → XGBoost classifier → malignancy probability
                        ↳ XGBoost regressor → survival estimate
                        ↳ probability threshold → stage
                        ↳ Gemini LLM → clinical narrative
```

## Methodology

1. **Visual branch** — ResNet-18-based `PulmoNetResNet`, pre-trained on public chest CT datasets (malignant vs. normal), then used to extract a 128-D embedding per patient per screening year.
2. **Clinical/temporal branch** — merges 5 NLST tables on `pid`/`study_yr` for 395 patients with complete 3-year histories; clinically-motivated imputation (nodules → 0, staging → 0, survival → max, other → median); synthetic cancer patients added (biologically-plausible growth simulation) to reach 616 patients.
3. **ARIMA trend extraction** — each of the 128 embedding dimensions is modeled as a 3-point series (`ARIMA(1,1,0)`), yielding 256 features (forecast + velocity).
4. **PCA compression** — 256 features → 10 principal components (57.3% variance explained) — this is the representation the deployed model actually uses.
5. **Fusion classifier** — XGBoost on the 10 PCA components + `age`/`gender`/`cigsmok` (leak-prone columns excluded).
6. **Staging** — a Random Forest staging model was trained but isn't currently called by the app, which uses a simple probability threshold instead.
7. **Survival** — a separate XGBoost regressor predicts cancer-free survival for cancer-positive patients.
8. **Narrative** — final results are sent to Gemini 2.5 Flash to generate a short clinical note.

## Model Performance

From the training notebook's saved outputs:

| Model | Task | Data | Result |
|---|---|---|---|
| PulmoNetResNet | Malignant vs. Normal | 1,672 train / 419 val imgs | **97.9%** best val accuracy |
| XGBoost, raw embeddings (5-fold CV) | Cancer vs. Healthy | 370 patients | **90.8%** val accuracy |
| **XGBoost + ARIMA + PCA (deployed)** | Cancer vs. Healthy | 616 patients | Train 97.8% / Test **75.8%** |
| Random Forest | Staging (Early vs. Advanced) | Cancer-positive subset | **78.4%** accuracy |
| XGBoost Regressor | Survival | Cancer-positive subset | MAE 0.16, **R² = −0.17** |

**Caveats**: the train/test gap on the deployed model suggests overfitting (small cohort + synthetic augmentation); the survival regressor's negative R² means it currently underperforms a naive mean-prediction baseline.

## Project Structure

```text
├── articles/                          # Research papers/docs
├── templates/                         # Flask HTML templates
├── Application.py                     # Flask backend + inference + LLM report
├── Preprocessing_CT.ipynb             # DICOM loading & normalization
├── data_preprocessed.ipynb            # NLST merging, imputation, labeling, splits
├── final_code_ResNet18_ARIMA_XGboost_PCA_.ipynb   # CNN + ARIMA + PCA + XGBoost training
├── best_pulmonet_standalone.pth       # Trained CNN weights
├── final_xgb_pca_model.pkl            # Deployed diagnostic classifier
├── final_xgb_survival_model.pkl       # Survival regressor
├── arima_scaler.pkl / arima_pca.pkl   # Fitted temporal-feature transforms
└── *.csv                              # Processed clinical tables
```

## Known Limitations

- Small cohort (395 real patients, 616 with synthetic augmentation)
- Overfitting on the deployed diagnostic model (train 97.8% vs. test 75.8%)
- Survival regressor is currently unreliable (negative R²)
- Single-scan inference approximates the temporal trend rather than computing it
- Notebooks contain hardcoded Windows paths — update before rerunning elsewhere
- Trained staging model isn't wired into the app

## Data Source

- **[National Lung Screening Trial (NLST)](https://cdas.cancer.gov/nlst/)**
- Public Kaggle chest CT datasets for CNN pre-training
