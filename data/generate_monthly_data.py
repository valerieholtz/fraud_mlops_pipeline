"""
Generate 12 monthly datasets with simulated data drift
from the PaySim synthetic fraud dataset.

Each month applies controlled changes:
- Scale transaction amounts upward (simulate inflation)
- Change distribution of transaction types (more TRANSFER in later months)
- Oversample fraud cases every 3rd month (simulate new fraud tactics)

Output:
data/fraud_month_1.csv ... data/fraud_month_12.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/fraud_transactions.csv")

def generate_monthly_datasets(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    for month in range(1, 13):
        df_month = df.sample(frac=0.1, random_state=month).copy()

        # 1. Amount scaling drift
        df_month["amount"] *= (1 + 0.02 * month)

        # 2. Change transaction mix
        if month % 2 == 0:
            mask = df_month["type"] == "PAYMENT"
            df_month.loc[mask, "type"] = "TRANSFER"

        # 3. Fraud ratio drift
        if month % 3 == 0:
            frauds = df_month[df_month["isFraud"] == 1]
            df_month = pd.concat([df_month, frauds], ignore_index=True)

        outfile = outdir / f"fraud_month_{month}.csv"
        df_month.to_csv(outfile, index=False)
        print(f"✅ Saved {outfile} with {len(df_month)} rows")

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    print(f"Loading baseline dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # Drop leakage columns
    drop_cols = [
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "nameOrig", "nameDest"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    generate_monthly_datasets(df, Path("data"))

if __name__ == "__main__":
    main()
