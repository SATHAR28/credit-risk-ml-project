import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def build_feature_frame(df: pd.DataFrame):
    required = [
        "gender",
        "no_of_days_employed",
        "credit_limit_used(%)",
        "credit_score",
        "default_in_last_6months",
        "occupation_type",
        "credit_card_default",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[required].copy()

    # Match web-app encoding: F -> 0, M -> 1
    work["gender"] = work["gender"].map({"F": 0, "M": 1})

    numeric_cols = [
        "gender",
        "no_of_days_employed",
        "credit_limit_used(%)",
        "credit_score",
        "default_in_last_6months",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["occupation_type"] = work["occupation_type"].fillna("Unknown").astype(str)

    occ_dummies = pd.get_dummies(work["occupation_type"], prefix="occupation")
    X = pd.concat([work[numeric_cols], occ_dummies], axis=1)
    y = work["credit_card_default"].astype(int)

    feature_names = X.columns.tolist()
    return X, y, feature_names


def build_model(random_state: int):
    base_estimators = [
        (
            "lr",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "et",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=350,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "knn",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", KNeighborsClassifier(n_neighbors=25)),
                ]
            ),
        ),
    ]

    final_estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Retrain credit risk model used by Flask app.")
    parser.add_argument("--train-csv", required=True, help="Path to train.csv")
    parser.add_argument("--model-out", required=True, help="Output path for model .pkl")
    parser.add_argument("--features-out", required=True, help="Output path for features.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    model_out = Path(args.model_out)
    features_out = Path(args.features_out)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training file not found: {train_csv}")

    df = pd.read_csv(train_csv)
    X, y, feature_names = build_feature_frame(df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(random_state=args.random_state)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_valid)
    val_proba = model.predict_proba(X_valid)[:, 1]

    print("Validation ROC-AUC:", round(float(roc_auc_score(y_valid, val_proba)), 5))
    print(classification_report(y_valid, val_pred, digits=4))

    # Fit on full training data before exporting
    model.fit(X, y)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    features_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    with open(features_out, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Saved model: {model_out}")
    print(f"Saved features: {features_out}")
    print(f"Feature count: {len(feature_names)}")
    print("Target distribution:", dict(zip(*np.unique(y, return_counts=True))))


if __name__ == "__main__":
    main()
