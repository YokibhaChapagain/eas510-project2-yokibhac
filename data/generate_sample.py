"""Generate a NEW, independently simulated OFFLINE demonstration dataset.


Usage: python data/generate_sample.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
RNG = np.random.default_rng(2026)
MONTHS = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
PLANS = ["Starter", "Pro", "Business", "Enterprise"]
PRICES = {"Starter": 50., "Pro": 120., "Business": 240., "Enterprise": 600.}


def generate():
    rows = []
    for i in range(750):
        customer = f"DEMO{i:05d}"
        plan = str(RNG.choice(PLANS, p=[0.4, 0.35, 0.2, 0.05]))
        base = float(RNG.uniform(8, 55))
        start = int(RNG.integers(0, 9))
        for tenure, month in enumerate(MONTHS[start:]):
            sessions = int(RNG.poisson(max(1, base + 4 * np.sin(tenure / 4) + RNG.normal(0, 4))))
            usage = float(np.clip(10 + 1.1 * sessions + RNG.normal(0, 12), 0, 100))
            payment_failures = int(RNG.binomial(2, 0.10))
            seats = int(max(1, RNG.poisson({"Starter": 2, "Pro": 4, "Business": 7,
                                            "Enterprise": 12}[plan])))
            support_tickets = int(RNG.poisson(0.6 + int(sessions < 15)))
            incident = int(RNG.random() < 0.06)
            nps_score = int(np.clip(round(4 + usage/20 + RNG.normal(0, 1.8)), 0, 10))
            hazard = float(np.clip(0.014 + 0.027 * (sessions < 18) +
                         0.016 * payment_failures + 0.015 * (usage < 35), 0.002, 0.17))
            churn = int(RNG.random() < hazard)
            rows.append({"user_id": customer, "month": month,
                         "tenure_month": tenure, "plan_type": plan,
                         "monthly_price": PRICES[plan], "mrr": PRICES[plan],
                         "sessions": sessions, "feature_usage_score": round(usage, 3),
                         "support_tickets": support_tickets,
                         "payment_failures": payment_failures, "churned": churn,
                         "nps_score": nps_score, "product_incident": incident,
                         "active_seats": seats})
            if churn:
                break
    df = pd.DataFrame(rows).sort_values(["user_id", "month"]).reset_index(drop=True)
    df["churned_next_month"] = (df.groupby("user_id")["churned"].shift(-1)
                                .fillna(0).astype(int))
    return df


def main():
    df = generate()
    last = df.month.max()
    eligible = df[(df.churned == 0) & (df.month < last)].copy()
    periods = {
        "train": eligible[eligible.month <= "2025-01-01"],
        "validation": eligible[eligible.month.between("2025-03-01", "2025-06-01")],
        "test": eligible[eligible.month.between("2025-08-01", "2025-11-01")],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"origin": "original independent simulation; NOT Hugging Face data",
                "seed": 2026, "model_evidence": False, "splits": {}}
    for name, part in periods.items():
        part = part.sort_values(["month", "user_id"]).reset_index(drop=True)
        if part.churned_next_month.nunique() != 2:
            raise RuntimeError(f"Demo {name} has only one class")
        part.to_csv(OUT / f"{name}.csv", index=False, date_format="%Y-%m-%d")
        manifest["splits"][name] = {"rows": len(part), "events": int(part.churned_next_month.sum()),
                                    "first_month": part.month.min().strftime("%Y-%m-%d"),
                                    "last_month": part.month.max().strftime("%Y-%m-%d")}
        print(f"{name}: {len(part):,} rows, {int(part.churned_next_month.sum())} churn events")
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
