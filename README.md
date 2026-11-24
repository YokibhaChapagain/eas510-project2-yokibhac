# Product Engagement and Subscription Behavior for SaaS Churn Prediction

**EAS 510 — The Basics of AI**  

**Research question:** Does product engagement information improve next-month SaaS churn prediction and customer-risk prioritization beyond traditional subscription and billing information?

This project investigates customer-month records from a **synthetic** B2B SaaS dataset. It compares subscription/billing, product-engagement, and combined feature groups. Instead of emphasizing accuracy alone, it evaluates a decision problem: *If a customer-success team can review only the top 20% of accounts by estimated churn risk each month, how many actual future churners are identified?*

## Data and source

- Source: [arti199919/synthetic-saas-churn-sample (Hugging Face)](https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample)
- Observation unit: one **customer-month**; target: `churned_next_month`.
- Original monthly table: **79,817** observations from **5,000** customers across Jan 2024–Dec 2025.
- Exclude rows already churned and the final month (future outcomes cannot be verified): **74,817** eligible records, including **1,056** next-month churn events (**1.41%**).
- Audits: zero missing values, zero duplicate customer-month keys; `mrr` duplicates `monthly_price` and is excluded from the model.

## Questions and preliminary hypotheses

1. **H1**: Usage signals add predictive information beyond subscription and billing information.
2. **H2**: Historical changes in usage may improve ranking beyond current engagement levels (not yet tested in the baseline notebook).
3. **H3**: Better predictive ranking can improve top-20%-capacity customer review compared with simple baselines.

## Approach and safeguards

**Feature groups:**

| Group | Included features |
|---|---|
| Billing | tenure, plan, monthly price, payment failures |
| Engagement | sessions, feature usage score, active seats |
| Combined | Billing + engagement |

**Models:** Dummy prior baseline, Logistic Regression, Decision Tree, Random Forest. Standardization and one-hot encoding are inside a scikit-learn `Pipeline` fitted on the training period only.

**Fixed calendar periods:** training Jan 2024–Jan 2025, validation Mar–Jun 2025, later evaluation Aug–Nov 2025; Feb and July act as gap months. The final observed month Dec 2025 is excluded. Model selection uses **validation Average Precision**. The later period was inspected during development and should **not** be described as a never-seen test set.

**Metrics:** Average Precision (AP) and per-month Recall@Top20% are central; Precision@Top20%, F1 at threshold 0.5, and Brier score are supplementary. Extremely rare churn means a 0.5 classifier threshold can generate all-negative predictions even when ranking is informative.

## Preliminary findings — original synthetic dataset, notebook protocol

> **Not sample/demo results.** These were calculated in `notebooks/01_data_exploration.ipynb` using the original Hugging Face data and the chronological periods above.

| Features | Later-period Average Precision | Later-period Recall@Top20% |
|---|---:|---:|
| Dummy prior | 0.016 | 21.8% |
| Billing | 0.031 | 38.0% |
| Engagement | 0.050 | 53.0% |
| Combined | **0.053** | **58.6%** |

Logistic Regression was selected for each feature group using validation Average Precision. Adding engagement to billing increased top-20% recall by approximately **20.6 percentage points** on this later period. Engagement alone performed much better than billing alone. Combined-model Precision@Top20% was **4.8%**, so most reviewed accounts would not actually churn—an important limitation.

These are **exploratory** associations in a simulated dataset, not evidence that an intervention will prevent churn. No causal effect of engagement, billing, or contact has been established.

## Environment and setup

Tested for the standalone CLI in **Python 3.13 / Linux CPU** with dependencies pinned in `requirements.txt`; also intended for local VS Code on macOS. No GPU, API key, or paid service is necessary for the offline demo. No required system packages beyond a working Python toolchain.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

**Default sample execution:** reads `data/sample/demo_customer_months.csv` without network access. It trains and evaluates models end-to-end, prints preliminary metrics, and writes results to `results/sample/cli_model_comparison.csv`, `cli_validation_model_comparison.csv`, and `cli_run_config.json`. A normal run should finish within five minutes on a typical CPU with 8 GB RAM; measure your own runtime and peak memory on the target environment before final submission. Scores can vary by dependency version; compare using the pinned versions and configuration in `cli_run_config.json` rather than requiring universal bitwise identity.

### Full-source evaluation (network required once for download)

```bash
python data/download.py
python main.py --full
```

### Notebooks

For exploratory charts and research commentary, open `notebooks/01_data_exploration.ipynb` in VS Code, selecting your environment as its kernel. Notebooks supplement, not replace, `main.py`.

## Project layout

```text
eas510-project2-<netid>/   # actual repository name has no space
├── README.md
├── report.md
├── main.py
├── config.py
├── requirements.txt
├── src/pipeline.py
├── data/download.py
├── data/generate_demo.py
├── data/sample/demo_customer_months.csv
├── notebooks/01_data_exploration.ipynb
├── results/sample/
├── results/full/
└── .gitignore
```

## Limitations and next steps

- The data are synthetic; results should not be generalized to real businesses without external validation.
- The same customers contribute observations in multiple periods; this models future months for known accounts, not performance on entirely unseen businesses.
- Multiple protocols were explored. Final reporting will choose one clearly defined primary comparison and document later-period inspection during development.
- Future work: Explore historical engagement trends, strengthen time-based validation, compare customer prioritization strategies, measure model uncertainty, and analyze prediction errors. If time permits, develop an intelligent agent that generates personalized retention summaries, explains customer churn risks, and suggests potential actions based on customer behavior.


## Contributions and AI assistance

Analysis decisions, code execution, dataset checks, and conclusions are the student's responsibility. AI assistance (ChatGPT) was used for project organization, code drafting, documentation, and debugging. Every reported result should be verified against the actually executed notebooks and saved CSV evidence. See `report.md` for the detailed statement.
