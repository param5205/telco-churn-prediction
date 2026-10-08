import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
     "TechSupport", "StreamingTV", "StreamingMovies",
]

# Data
def load_and_clean(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(9.9)
    df = df.drop(columns=["customerID"])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_addons"]= (df[ADDON_COLS] =="Yes" ).sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include= "number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        sparse_threshold=0.0
    )


#Plots

def save_eda_plots(df: pd.DataFrame, out_dir: Path)-> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    df[TARGET].value_counts(normalize=True).rename({0: "Stayed", 1:"Churned"}).plot.bar(ax=axes[0])
    axes[0].set_title("Class balance")
    axes[0].set_ylabel("Share of customers")

    df.groupby("Contract")[TARGET].mean().plot.bar(ax=axes[1], color="tab:orange")
    axes[1].set_title("Churn rate by contract type")
    axes[1].set_ylabel("Churn rate")

    buckets=pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    df.groupby(buckets, observed=True)[TARGET].mean().plot.bar(ax=axes[2], color="tab:green")
    axes[2].set_title("Churn rate by tenure (months)")
    axes[2].set_ylabel("Churn rate")


    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir/ "eda_overview.png", dpi=150)
    plt.close(fig)


def save_evaluation_plots(model, X_test, y_test, out_dir: Path) -> None:
    fig, ax=plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_feature_importance(model, X_test, y_test, out_dir: Path, top_n: int= 10) -> pd.DataFrame:
    # Permuatation importance on the raw column (works on the whole pipeline).
    result= permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp= (
        pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top = imp.head(top_n).iloc[::-1]
    fig, ax=plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    return imp


# Modeling

def get_models() -> dict:
    return{
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"), "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

PARAM_GRIDS= {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [1, 5, 10]},
    "Gradient Boosting": {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [3, 5, None]},
}


def compare_models(preprocessor, X_train, y_train, cv) -> pd.DataFrame:
    scoring= {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"}
    rows = []
    for name, clf in get_models().items():
        pipe= Pipeline([("prep", preprocessor), ( "clf", clf)])
        scores= cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name, **{m: scores[f"test_{m}"].mean () for m in scoring}})
        print(f" {name:<20} ROC-AUC={rows[-1]['roc_auc']:.3f} F1={rows[-1]['f1']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

def threshold_table(y_true, proba)-> pd.DataFrame:
    #Precison/recall trade-off at different descion threshold.
    rows= []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score ( y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred),
                "f1": f1_score(y_true, pred),
                "customers_flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


# Main

def main(data_path: Path, out_root: Path) -> None:
    fig_dir= out_root / "reports" / "figures"
    model_dir= out_root / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df= add_features(load_and_clean(data_path))
    print(f" {df.shape[0]} rows, churn rate= {df[TARGET].mean():.1%}")
    save_eda_plots(df, fig_dir)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor= build_preprocessor(X_train)

    print("Comparing models (5-fold CV on training set)...")
    comparison= compare_models(preprocessor, X_train, y_train, cv)
    comparison.to_csv(out_root/ "reports" / "model_comparison.csv", index=False)
    best_name = comparison.loc[0, "model"]

    print(f" Tuning best model: {best_name}...")
    pipe=Pipeline([("prep", preprocessor), ("clf", get_models()[best_name])])
    search= GridSearchCV(pipe, PARAM_GRIDS[best_name], scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    model= search.best_estimator_
    print(f" best params: {search.best_params_} (CV ROC-AUC={ search.best_score_:.3f})")
    
    print("Evaluating on held-out test set...")
    proba=model.predict_proba(X_test)[:,1]
    pred= (proba >=0.5).astype(int)
    metrics = {
        "best_model": best_name,
        "best_params": {k: (None if v is None else v) for k, v in search.best_params_.items()},
        "cv_roc_auc": round(float(search.best_score_), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_f1": round(float(f1_score(y_test, pred)), 4),
        "test_precision": round(float(precision_score(y_test, pred)), 4),
        "test_recall": round(float(recall_score(y_test, pred)), 4),
    }
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))
    threshold_table(y_test, proba).to_csv(out_root / "reports" / "threshold_analysis.csv", index=False)

    save_evaluation_plots(model, X_test, y_test, fig_dir)
    importance = save_feature_importance(model, X_test, y_test, fig_dir)
    importance.to_csv(out_root / "reports" / "feature_importance.csv", index=False)

    with open(out_root / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, model_dir / "churn_model.joblib")
    print(json.dumps(metrics, indent=2))
    print(f"Done. Reports in {out_root / 'reports'}, model in {model_dir}")


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="Train a customer churn model.")
    parser.add_argument("--data", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    parser.add_argument("--out", type=Path, default=Path("."), help="Project root for reports/ and models/")
    args = parser.parse_args()
    main(args.data, args.out)