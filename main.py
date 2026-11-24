"""EAS 510 required CLI. Default: offline bundled demo; --full: source dataset."""
import argparse
import json
import pandas as pd

from config import (PROJECT_ROOT, RAW_DATA, SAMPLE_FILES, SEED,
                    FEATURE_GROUPS, TOP_FRACTION)
from src.pipeline import prepare, split_periods, run_comparison


def main():
    parser = argparse.ArgumentParser(description="SaaS next-month churn experiment")
    parser.add_argument("--full", action="store_true", help="Run on downloaded Hugging Face data")
    args = parser.parse_args()
    mode = "full" if args.full else "sample"

    if args.full:
        if not RAW_DATA.is_file():
            parser.error("Full data missing. First run: python data/download.py")
        raw = pd.read_parquet(RAW_DATA)
        eligible = prepare(raw)
        train, validation, test = split_periods(eligible)
        source = str(RAW_DATA.relative_to(PROJECT_ROOT))
    else:
        missing = [str(p) for p in SAMPLE_FILES.values() if not p.is_file()]
        if missing:
            parser.error(f"Bundled sample missing: {missing}. See data/sample/README.md")
        train, validation, test = (pd.read_csv(SAMPLE_FILES[n], parse_dates=["month"])
                                   for n in ("train", "validation", "test"))
        eligible = pd.concat([train, validation, test], ignore_index=True)
        raw = eligible
        source = "data/sample/{train,validation,test}.csv (independent synthetic demo)"

    val_table, test_table = run_comparison(train, validation, test)
    out_dir = PROJECT_ROOT / "results" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    val_table.to_csv(out_dir / "cli_validation_model_comparison.csv", index=False)
    test_table.to_csv(out_dir / "cli_model_comparison.csv", index=False)
    config = {
        "mode": mode, "source": source, "seed": SEED,
        "top_fraction": TOP_FRACTION, "feature_groups": FEATURE_GROUPS,
        "raw_rows": len(raw), "eligible_rows": len(eligible),
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_months": {
            k: [str(df.month.min().date()), str(df.month.max().date())]
            for k, df in (("train", train), ("validation", validation), ("test", test))
        }
    }
    (out_dir / "cli_run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"MODE: {mode} | {len(eligible):,} eligible rows")
    print("Model selection: validation AP; later period is evaluated separately.")
    print(test_table[["features", "model", "average_precision", "recall_at_top20",
                      "precision_at_top20"]].round(4).to_string(index=False))
    print(f"Saved: {out_dir}")
    if not args.full:
        print("IMPORTANT: independent demo only. Do not use these numbers as source-data results.")


if __name__ == "__main__":
    main()
