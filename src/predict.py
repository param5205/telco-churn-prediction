import argparse
from pathlib import Path

import joblib
import pandas as pd

from train import add_features

def main(model_path: Path, input_path: Path, output_path: Path) -> None:
    model=joblib.load(model_path)
    df=pd.read_csv(input_path)
    ids=df["customerID"] if "customerID" in df else pd.Series(range(len(df)), name="row")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    X= add_features(df.drop(columns=[c for c in ["customerID", "Churn"] if c in df]))
    out = pd.DataFrame({"customer": ids, "churn_probability": model.predict_proba(X)[:,1].round(4)})
    out.sort_values("churn_probability", ascending=False).to_csv(output_path, index=False)
    print(f"Saved {len(out)} predicyions to {output_path}")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model/churn_model.joblib"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    a= p.parse_args()
    main(a.model, a.input.a.output)