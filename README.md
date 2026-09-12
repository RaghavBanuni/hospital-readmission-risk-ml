# 30-Day Hospital Readmission Risk

A readmission model built the way a hospital would have to use it: risk scores
ranked against a **fixed care-transition team capacity**, probabilities that are
calibrated, and a **subgroup audit** that reports where the model is worse for
some patients than for others.

## Overview

Unplanned readmission within 30 days is penalised by payers and is a genuine
harm to patients. A model that helps has to answer an operational question, not a
leaderboard one: *of the patients discharged this week, which 40 should the
transition-care nurses call first?*

That framing drives three design choices most readmission notebooks skip:

1. **Capacity-aware evaluation.** Precision and recall are reported at the top
   *k*% of discharges, because the intervention team can only reach so many
   people. Sensitivity at an unreachable threshold is decoration.
2. **Calibration is a first-class metric.** Clinicians and utilisation review
   compare risk against thresholds ("above 20%"), so a 20% prediction has to mean
   20%. Brier score, reliability bins and calibration-in-the-large are reported
   before and after isotonic calibration.
3. **A fairness audit, not a fairness slogan.** Performance is broken out by age
   band, sex, insurance type and language preference, and the report shows the
   **gaps**: false-negative rate difference (equal-opportunity gap), selection-rate
   difference (demographic parity gap) and per-group calibration error. A model
   that misses high-risk patients in one group more often than another is not
   deployable, however good the overall AUC.

## Domain context

The features are the ones a discharge planner actually has on the day of
discharge - which is also what keeps the model honest about leakage:

| group | features |
| --- | --- |
| demographics | age, sex, insurance, preferred language, lives alone |
| history | prior admissions in 12 months, prior ED visits, days since last discharge |
| this admission | length of stay, admission type (emergency/urgent/elective), ICU hours, number of procedures |
| clinical burden | comorbidity count, diabetes, heart failure, COPD, renal disease, medication count, polypharmacy flag |
| labs at discharge | haemoglobin, sodium, creatinine, albumin (with realistic missingness) |
| discharge | disposition (home / home health / skilled nursing), discharge against advice, follow-up appointment booked |

**No post-discharge information is used**, because anything observed after the
patient leaves would not exist at scoring time. The feature list is enforced in
code by a contract object rather than by discipline.

## Data

Synthetic and generated in-repo (`readmission/data.py`) from an explicit log-odds
model with documented coefficients - no protected health information, nothing
redistributed, and the true drivers are known so the tests can assert that the
model recovers them. Deliberate realism:

- **~15% readmission rate**, so accuracy is a useless metric by construction;
- **missing labs** (albumin missing far more often than sodium), because real
  discharge data is incomplete and the pipeline must impute;
- **interaction effects**: heart failure plus a missed follow-up appointment is
  worse than either alone;
- **subgroup structure**: a lower baseline follow-up-booking rate for one
  insurance group creates a genuine disparity for the audit to find, which is the
  point - a fairness report that always says "no gap" tests nothing.

## Methodology

- **Split**: stratified train/validation/test (60/20/20). The threshold and the
  champion are chosen on validation; the test split is scored once.
- **Models**: class-weighted logistic regression (the interpretable baseline that
  a clinical committee can read) versus histogram gradient boosting, both inside a
  `Pipeline` with median imputation, missingness indicators, scaling and one-hot
  encoding.
- **Calibration**: cross-fitted isotonic regression on the training rows only.
- **Metrics**: ROC-AUC, average precision, Brier, expected calibration error,
  recall/precision at capacity, lift, and **net benefit** at the operating
  threshold given the cost of an outreach call versus the cost of a readmission.
- **Explainability**: standardised logistic coefficients and permutation
  importance for the boosted model, reported side by side - agreement between them
  is evidence the model learned clinical signal rather than noise.

## Project structure

```
readmission/
  data.py       synthetic encounter generator with documented log-odds drivers
  features.py   feature contract, coercion, preprocessing pipeline
  model.py      candidate models, calibration, permutation importance
  evaluate.py   discrimination, calibration, capacity metrics, net benefit
  fairness.py   subgroup metrics and gap summaries
  cli.py        data / train / audit commands
tests/          generator, features, evaluation and fairness tests
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# inspect the cohort and its readmission rates
python -m readmission.cli data --patients 12000 --out data/encounters.csv

# train, calibrate, choose an operating point at 8% outreach capacity
python -m readmission.cli train --patients 30000 --capacity 0.08 --out reports/

# fairness audit of the trained champion
python -m readmission.cli audit --patients 30000 --capacity 0.08 --out reports/

pytest -q
```

Outputs: `model_comparison.csv`, `calibration.csv`, `capacity_metrics.csv`,
`net_benefit.csv`, `feature_importance.csv`, `subgroup_metrics.csv`,
`fairness_gaps.csv` and `training_report.json`. No metric is quoted in this README
on purpose - the numbers depend on the seed and the capacity you configure, and
the CLI prints them.

## Design decisions and trade-offs

- **Why keep the logistic model?** Clinical governance committees approve models
  they can read. If boosting cannot beat a transparent baseline on validation
  average precision, the baseline wins.
- **Why isotonic rather than Platt?** Enough training rows here for the flexible
  monotone fit; the function exposes `method` so a smaller cohort can use sigmoid.
- **Why report gaps rather than one fairness number?** "Fair" is not a single
  scalar: demographic parity and equal opportunity conflict whenever base rates
  differ between groups. The audit reports both and names the trade-off instead of
  pretending it away.
- **Why not resample to balance the classes?** Oversampling distorts the
  probability scale, and calibrated probabilities are the whole point here. Class
  weights achieve the same objective without inventing patients.
- **What this is not:** a clinical decision support system, validated on real
  patients, or fit for care decisions. It is a methodology demonstration on
  synthetic data.

## Skills demonstrated

Clinical problem framing, leakage-aware feature design, imbalanced classification,
probability calibration, capacity- and cost-aware decision analysis, algorithmic
fairness auditing (equal opportunity, demographic parity, per-group calibration),
scikit-learn pipeline engineering, permutation importance, reproducible synthetic
data generation, pytest.

## Possible extensions

Competing-risk survival modelling (time to readmission with death as a competing
event), threshold optimisation per subgroup with an explicit equity constraint,
SHAP value reporting, and drift monitoring across admission months.

## License

MIT - see `LICENSE`.
