import pandas as pd
import os

data_dir = "archive_nlst"

datasets = [
    "nlst_780_canc_idc_20210527.csv",
    "nlst_780_ctab_idc_20210527.csv",
    "nlst_780_ctabc_idc_20210527.csv",
    "nlst_780_prsn_idc_20210527.csv",
    "nlst_780_screen_idc_20210527.csv"
]

for ds in datasets:
    path = os.path.join(data_dir, ds)
    print(f"--- Reading {ds} ---")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(df.head(2))
    print("\n")
