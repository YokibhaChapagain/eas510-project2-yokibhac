"""Reproducible EAS 510 customer-month churn experiment settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42
TARGET = "churned_next_month"
DATASET_ID = "arti199919/synthetic-saas-churn-sample"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA = RAW_DIR / "user_monthly.parquet"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_FILES = {
    "train": SAMPLE_DIR / "train.csv",
    "validation": SAMPLE_DIR / "validation.csv",
    "test": SAMPLE_DIR / "test.csv",
}
BILLING_FEATURES = ["tenure_month", "plan_type", "monthly_price", "payment_failures"]
ENGAGEMENT_FEATURES = ["sessions", "feature_usage_score", "active_seats"]
FEATURE_GROUPS = {
    "billing": BILLING_FEATURES,
    "engagement": ENGAGEMENT_FEATURES,
    "combined": BILLING_FEATURES + ENGAGEMENT_FEATURES,
}
TRAIN_END = "2025-01-01"
VALIDATION_START, VALIDATION_END = "2025-03-01", "2025-06-01"
TEST_START, TEST_END = "2025-08-01", "2025-11-01"
TOP_FRACTION = 0.20
