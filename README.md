# SSI prediction after open surgery for cholelithiasis

Core analysis code for **A parsimonious machine-learning model for peri-discharge stratification of surgical site infection after open surgery for cholelithiasis: a single-centre retrospective cohort study**.

## Files

All scripts are in `analysis/` and are listed in execution order:

| Script | Purpose |
| --- | --- |
| `select_models.py` | Compare model families, imbalance strategies, and hyperparameters |
| `evaluate_full_models.py` | Evaluate the selected full-predictor configurations |
| `compute_feature_importance.py` | Estimate permutation and model-specific importance |
| `screen_features.py` | Screen contributions and cross-validation stability |
| `rank_features.py` | Calculate consensus feature rankings |
| `evaluate_feature_subsets.py` | Evaluate ranked predictor subsets |
| `train_final_model.py` | Retune, calibrate, and evaluate the final model |
| `run_ablation.py` | Compare feature-selection strategies |
| `compare_reduced_models.py` | Compare models on the reduced predictor set |
| `compare_reclassification.py` | Calculate corrected final-versus-full NRI and IDI |

`feature_schema.csv` contains predictor names and computational types. Figure-generation and table-formatting scripts are omitted.

## Restricted inputs

Place authorized study inputs under `analysis/data/`:

- `clinical_data.csv`: cleaned data with unique `ID`, binary `Infection` (0/1), and the 64 predictors in the schema. Blank entries indicate missing values.
- `split_folds.csv`: original fixed assignments with `ID,split,fold_id,Infection`; `split` is DEV or VAL, and DEV folds are 0–4.

Use the original 348-patient analysis list (DEV 278; VAL 70, including 11 SSI events). Records outside that list are excluded. Do not regenerate the fixed assignments. Clinical units and category definitions must be obtained from the data custodian. Patient data, assignments, and individual predictions are not included in this repository.

## Execution

For model development, use Python 3.13.13 and the recorded core dependencies:

```text
python -m pip install -r requirements.txt
cd analysis
python select_models.py
python evaluate_full_models.py
python compute_feature_importance.py
python screen_features.py
python rank_features.py
python evaluate_feature_subsets.py
python train_final_model.py
python run_ablation.py
python compare_reduced_models.py
```

Run these steps from `analysis/`; generated files are written under `analysis/outputs/`. Scheme2 is the primary mean-based screening procedure. Scheme1 is the sensitivity procedure, using median-based screening and median permutation importance, with mean internal importance in consensus scoring.

For corrected NRI/IDI, use a separate Python 3.12.14 environment. From the repository root:

```text
python -m pip install -r requirements-reclassification.txt
python analysis/compare_reclassification.py --root analysis --out analysis/outputs/reclassification
```

The comparison reads `outputs/final_model/Final_Predictions_CatBoost+S0_Top4.csv` (`ID,y_true,prob_uncalib`) and `outputs/full_models/CatBoost+S0/indep_predictions.csv` (`ID,y_true,pred_prob`) under the analysis root, checking IDs and outcomes against `data/split_folds.csv`. It uses original-sample point estimates, 1,000 paired bootstrap draws with `default_rng(42)`, percentile 95% intervals, and normal-approximation two-sided p values based on bootstrap standard errors.

Expected results: NRI −0.173 (95% CI −0.738 to 0.400; p=0.557); IDI −0.007 (−0.079 to 0.068; p=0.845).

## Interpretation and verification

The final and full models both use depth 3 and 300 iterations, with learning rates 0.03 and 0.1, respectively. Ablation uses 600 iterations and is not the reclassification input. These comparisons do not isolate predictor removal or establish equivalence. Calibrated operating thresholds use fitted DEV ensemble probabilities, not strictly out-of-fold probabilities. Drain time accumulates through discharge; this analysis concerns exploratory peri-discharge stratification of observed SSI status.

Source checks and reclassification against saved predictions were completed; the full training grid was not rerun during code preparation. Reproducing study results requires the restricted inputs and original assignments. Generated patient-level outputs must remain in controlled storage.
