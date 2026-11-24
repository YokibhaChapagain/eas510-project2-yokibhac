"""Train and evaluate customer-month churn models with chronological boundaries.

The primary scientific results refer to the ORIGINAL Hugging Face dataset. The
bundled demo illustrates the SAME workflow on separately generated records.
"""
import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (SEED, TARGET, FEATURE_GROUPS, TOP_FRACTION,
                    TRAIN_END, VALIDATION_START, VALIDATION_END,
                    TEST_START, TEST_END)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only active customer-months with observable next-month outcomes."""
    df = raw.copy()
    df["month"] = pd.to_datetime(df["month"])
    required = {"user_id", "month", "churned", TARGET, *FEATURE_GROUPS["combined"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.duplicated(["user_id", "month"]).any():
        raise ValueError("Duplicate customer-month records")
    if df[TARGET].isna().any():
        raise ValueError("Missing target values")
    last_month = df["month"].max()
    df = df.loc[(df["churned"] == 0) & (df["month"] < last_month)].copy()
    return df.sort_values(["month", "user_id"]).reset_index(drop=True)


def split_periods(eligible: pd.DataFrame):
    """Use gaps before validation and test so future-label windows don't overlap."""
    train = eligible.loc[eligible.month <= TRAIN_END].copy()
    val = eligible.loc[eligible.month.between(VALIDATION_START, VALIDATION_END)].copy()
    test = eligible.loc[eligible.month.between(TEST_START, TEST_END)].copy()
    return train, val, test


def check_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    parts = {"train": train, "validation": validation, "test": test}
    full = pd.concat(parts.values(), ignore_index=True)
    if full.duplicated(["user_id", "month"]).any():
        raise ValueError("Duplicate customer-month records across splits")
    bounds = {
        "train": (None, pd.Timestamp(TRAIN_END)),
        "validation": (pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END)),
        "test": (pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
    }
    for name, df in parts.items():
        required = {"user_id", "month", TARGET, *FEATURE_GROUPS["combined"]}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if df.empty or df[TARGET].nunique() < 2:
            raise ValueError(f"{name} must contain two target classes")
        lo, hi = bounds[name]
        if lo is not None and df["month"].min() < lo:
            raise ValueError(f"{name} begins before expected date")
        if df["month"].max() > hi:
            raise ValueError(f"{name} extends beyond expected date")
        if "churned" in df and (df["churned"] != 0).any():
            raise ValueError(f"{name} contains already-churned customer-months")
    if not (train.month.max() < validation.month.min() < test.month.min()):
        raise ValueError("Splits are not chronological")


def make_pipeline(columns, estimator):
    numeric = [c for c in columns if c != "plan_type"]
    transforms = [("numeric", StandardScaler(), numeric)]
    if "plan_type" in columns:
        transforms.append(("plan", OneHotEncoder(handle_unknown="ignore"), ["plan_type"]))
    return Pipeline([
        ("preprocess", ColumnTransformer(transforms)),
        ("classifier", clone(estimator)),
    ])


def model_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                                   min_samples_leaf=20, random_state=SEED, n_jobs=1),
    }


def evaluate(data: pd.DataFrame, scores):
    """Compute AP and select the top 20% *within each month* (capacity per month)."""
    df = data.reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid probability vector")
    chosen = np.zeros(len(df), dtype=bool)
    for month, idx in df.groupby("month").indices.items():
        idx = np.asarray(idx)
        n = math.ceil(len(idx) * TOP_FRACTION)
        # Use a seeded tie breaker for equally ranked accounts.
        rnd = np.random.default_rng(SEED + int(pd.Timestamp(month).month) + int(pd.Timestamp(month).year))
        ranked = np.lexsort((rnd.random(len(idx)), -scores[idx]))
        chosen[idx[ranked[:n]]] = True
    captured = int(np.count_nonzero(chosen & (y == 1)))
    positives = int(y.sum())
    predictions = (scores >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "recall_at_top20": (captured / positives) if positives else np.nan,
        "precision_at_top20": captured / int(chosen.sum()),
        "captured_churners": captured,
        "total_churners": positives,
        "selected_customer_months": int(chosen.sum()),
        "f1_at_threshold_0_5": float(f1_score(y, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y, scores)),
    }


def run_comparison(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    """Fit on train, select models by validation AP, evaluate once on later test."""
    for data in (train, validation, test):
        data["month"] = pd.to_datetime(data["month"])
    check_splits(train, validation, test)
    fitted = {}
    validations = []
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train[["tenure_month"]], train[TARGET])
    validations.append({"features": "none", "model": "Dummy prior",
        **evaluate(validation, dummy.predict_proba(validation[["tenure_month"]])[:, 1])})
    for group, columns in FEATURE_GROUPS.items():
        for model_name, estimator in model_candidates().items():
            model = make_pipeline(columns, estimator)
            model.fit(train[columns], train[TARGET])
            fitted[group, model_name] = model
            validations.append({"features": group, "model": model_name,
                **evaluate(validation, model.predict_proba(validation[columns])[:, 1])})
    val_table = pd.DataFrame(validations)
    winners = (val_table[val_table.features != "none"]
               .sort_values(["average_precision", "model"], ascending=[False, True])
               .drop_duplicates("features"))
    tests = [{"features": "none", "model": "Dummy prior",
              **evaluate(test, dummy.predict_proba(test[["tenure_month"]])[:, 1])}]
    for winner in winners.itertuples(index=False):
        cols = FEATURE_GROUPS[winner.features]
        model = fitted[winner.features, winner.model]
        tests.append({"features": winner.features, "model": winner.model,
            **evaluate(test, model.predict_proba(test[cols])[:, 1])})
    return val_table, pd.DataFrame(tests)
