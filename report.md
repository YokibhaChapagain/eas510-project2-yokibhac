# Predicting SaaS Customer Churn Using Product Engagement and Subscription Behavior

**EAS 510 Project 2 — Working Report (not final)**  
**Author:** Yokibha Chapagain  
**Status:** Preliminary results, November 23, 2026

## 1. Introduction

**Research question.** Does product engagement and its changing patterns improve next-month SaaS churn prediction relative to using subscription and billing features alone? Can a higher-quality ranking support resource-constrained retention review?

## 2. Background and related work

Sanchez Ramirez et al. (2024) demonstrated the value of incorporating product usage data into B2B churn prediction. Sanches et al. (2025) explored machine learning approaches for SaaS churn prediction, while Kotan et al. (2025) investigated feature selection to improve predictive performance.

Building on these studies, this project compares subscription and product engagement features, focusing on whether their combination improves churn prediction and supports more effective customer-retention prioritization.
## 3. Data and initial quality checks

The [Synthetic B2B SaaS Churn & Revenue Sample](https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample) includes 79,817 customer-month records and 5,000 distinct customers from January 2024 to December 2025. After excluding accounts that had already churned and the final observation month, 74,817 rows are eligible, including 1,056 next-month churn events (1.41%). No missing values or duplicate customer-month keys were detected. The price and MRR columns are identical in all records and therefore the redundant MRR column is excluded.

**Limitation:** This is simulated data. The data card specifies license `other`; redistribution terms for the original files have not been established. The independently generated demo sample is not a source-data subset and may require instructor approval under course rules.

## 4. Methodology (baseline protocol)

We compare billing features (tenure, subscription plan, price, payment failures), engagement features (sessions, feature usage, active seats), and the combined feature group. Logistic Regression, Decision Tree, and Random Forest are benchmarked against a DummyClassifier prior baseline. Preprocessing is embedded in model pipelines and learned only from training records. Chronological training, validation, and later test periods are used; model selection is based on validation Average Precision. Metrics include Average Precision and per-month Recall@Top20%.


## 5. Preliminary results (later period Aug–Nov 2025)

| Features | Model selected on validation | Average Precision | Recall@Top20% |
|---|---|---:|---:|
| None | Dummy prior | 0.016 | 21.8% |
| Billing | Logistic Regression | 0.031 | 38.0% |
| Engagement | Logistic Regression | 0.050 | 53.0% |
| Combined | Logistic Regression | **0.053** | **58.6%** |

These preliminary scores were read from the executed exploratory notebook. On the chosen later period, engagement adds materially to billing-only ranking, and combining both groups adds a smaller benefit beyond engagement alone. The combined model has Precision@Top20% of 4.8%. The ordinary 0.5 classification threshold predicts no churn events; therefore binary-classification F1 is zero even though risk ranking is informative.

Evidence: [exploration notebook](notebooks/01_data_exploration.ipynb), [model comparison CSV](results/full/test_model_comparison.csv), [figure](results/full/test_model_comparison.png).

## 6. Discussion and limitations

Preliminary evidence indicates that product engagement is associated with subsequent churn, but this does not establish causality or show that contacting customers prevents churn. Rare churn (about 1.41% of eligible records), repeated observations per customer, synthetic data, time-boundary decisions, and inspection of the later period limit generalizability. Develop explicit uncertainty estimates and compare against transparent ranking rules in future work.

## 7. Conclusion

_To complete using the final, verified evaluation protocol._

## 8. References

- arti199919. *Synthetic B2B SaaS Churn & Revenue (Sample).* Hugging Face. https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample


## 9. Contributions and AI assistance

ChatGPT assisted with organization, code drafting, debugging, and documentation. The student is responsible for executing, examining, and independently explaining all results. All generated text and code should be verified before submission. _Expand this section accurately to describe actual use._
