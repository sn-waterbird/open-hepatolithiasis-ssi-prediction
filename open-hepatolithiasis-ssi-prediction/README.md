# SSI prediction after open surgery for cholelithiasis

Research code accompanying **A parsimonious machine-learning model for peri-discharge stratification of surgical site infection after open surgery for cholelithiasis: a single-centre retrospective cohort study**. The workflow compares four model families and four imbalance strategies, applies multidimensional consensus feature selection (MCFS), and evaluates a four-predictor CatBoost model.

This repository contains source code, environment specifications, a feature schema, and aggregate reclassification results. Patient-level data, patient identifiers, fixed patient/fold assignments, individual predictions, and trained models are not distributed. Running the study analyses requires authorized access to the clinical dataset and the original fixed split. No synthetic dataset is presented as study data.

## Contents

| Path | Purpose |
| --- | --- |
| `analysis/` | Original training, feature selection, evaluation, table, and figure scripts |
| `analysis/reclassification.py` | Corrected final-model paired NRI/IDI calculation |
| `analysis/figures/figS1_final.py` | Revised Supplementary Figure S1 |
| `docs/REPRODUCIBILITY.md` | Input contract, execution order, and interpretation limits |
| `docs/feature_schema.csv` | Predictor names and computational types; no patient values |
| `requirements.txt` | Original analysis dependencies |
| `requirements-reclassification.txt` | Revised NRI/IDI and figure dependencies |
| `tests/` | Synthetic-data checks of NRI/IDI formulas and pairing safeguards |
| `results/reported_reclassification.json` | Aggregate results for the manuscript comparison |

## Start here

Read [the reproduction guide](docs/REPRODUCIBILITY.md) before running any training script. Preserve the original UTF-8 filenames and directory structure, and run original scripts from `analysis/`.

For the data-free reclassification checks, create a Python 3.12.14 environment, install `requirements-reclassification.txt`, and run from the repository root:

```text
python -m pip install -r requirements-reclassification.txt
python -m unittest discover -s tests -v
python analysis/reclassification.py --help
```

For the original model workflow, use a separate Python 3.13.13 environment and `requirements.txt`. The full model grid is computationally intensive. Environment versions are recorded from the saved study environment; the complete training workflow was not rerun during repository packaging.

## Scope and interpretation

The final cohort contains 348 patients (DEV 278; VAL 70). The four clinical predictors are drain time, history of abdominal surgery, monocyte ratio, and surgical duration. Drain time accumulates through discharge; the analysis concerns exploratory peri-discharge stratification of observed SSI status. This code is research material, not a validated clinical decision service.

SHAP displays describe the uncalibrated four-predictor model in DEV. The corrected NRI/IDI comparison uses uncalibrated final four-predictor versus full 64-predictor probabilities in VAL. The full and reduced models have separately selected hyperparameters; the comparison does not isolate predictor removal or establish equivalence.

## Data and reuse

Clinical data are not included. Requests for access must follow the authors' institutional and ethics requirements; public availability of code does not grant access to patient data. This package does not supply missing clinical category definitions or laboratory units.

No reuse license has been selected by the authors in this release. Public visibility alone is not an open-source license; contact the authors regarding reuse. Publication DOI and formal citation metadata can be added when available.
