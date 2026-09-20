# Reproduction guide

## Inputs and fixed analysis population

Work from `analysis/`. Authorized users must supply:

1. `开腹肝胆道结石手术SSI清洗.csv`: UTF-8 CSV with unique `ID`, binary `Infection` (0/1), and the 64 columns in `feature_schema.csv`. Blank fields represent missing values. An optional raw source named `开腹肝胆道结石手术SSI数据.csv` can be processed with `数据清理.py`.
2. `splits/split_folds_full.csv`: the **original fixed** patient assignments, with `ID`, `split` (`DEV` or `VAL`), `fold_id` (DEV folds 0–4; VAL uses the saved sentinel), and `Infection`. Outcome labels must agree with the cleaned data.
3. `特征类别/feature_types_summary.csv`: supplied schema with `feature,type` only; no observed-value previews are published.

The analytic population is the 348 IDs in the original fixed split, including 66 SSI events. The original cleaned source may contain 353 rows; the five records outside the fixed list are not analysis subjects. Use the fixed IDs consistently. Do not reconstruct a split from a random seed and treat it as the original assignment. No split-generation script is included in the study reproduction sequence.

The schema records computational types, not a complete clinical data dictionary. Clinical units, category meanings, and collection timing must be obtained from the study data custodian; no inferred definitions are supplied here. `特征类别.py` is retained as provenance of type inference but should not replace the supplied schema during exact reproduction.

## Model development environment and execution order

Use Python 3.13.13 and install `../requirements.txt`. The pinned package versions match saved environment metadata and the manuscript. `openpyxl` is an additional dependency for optional table export; its historical version was not recorded. Use UTF-8 filenames and run the following commands from `analysis/`, in order:

```text
python validate_inputs.py
python 训练全特征.py
python 训练全特征输出.py
python 特征重要性.py
python 特征筛选.py
python 特征共识.py
python 遍历训练.py
python 最终训练.py
python 特征选择消融.py
python 精简特征重比较.py
```

The scripts retain their original calculations and relative output directories. `docs/source_manifest.json` records hashes of unchanged source files. The initial full-model grid creates `训练全特征/scheme_to_best_params.csv`; downstream steps depend on it. Feature ranking uses `特征筛选共识/CatBoost+S0/contribution_scheme2.csv`.

The final manuscript uses **scheme2 (mean-based screening)** as its primary analysis and scheme1 for sensitivity analysis. Historical comments in the original selection script label these roles differently; the configured calls and the final manuscript determine their actual role. Scheme1 uses median positive-contribution screening and median PI but mean PVC/LFC in consensus scoring; it is not an all-median aggregation.

After training, optional table/figure generation is:

```text
python Table/特征基线.py
python figures/fig2_mcfs_integrated.py
python figures/fig3_model_performance.py
python figures/fig4_calibration_dca.py
python figures/fig5_shap.py
python figures/figS2_ablation.py
```

The study-flow diagram is an author-prepared illustration and has no generating script in this package. Tables and exported figures may require journal-specific formatting.

## Corrected final-model NRI and IDI

Use a separate Python 3.12.14 environment with `requirements-reclassification.txt`. From the repository root:

```text
python analysis/reclassification.py --root analysis --out analysis/NRI_IDI分析
python analysis/figures/figS1_final.py --input-dir analysis/NRI_IDI分析
```

Required inputs under `--root` are:

- `最终训练/Final_Predictions_CatBoost+S0_Top4.csv`: `ID,y_true,prob_uncalib` (additional calibrated columns are ignored).
- `训练全特征输出/CatBoost+S0/indep_predictions.csv`: `ID,y_true,pred_prob`.
- `splits/split_folds_full.csv`: fixed VAL membership and outcomes.

Rows are joined by ID. Both outcome and membership must match, with 70 VAL patients and 11 events. Bootstrap row order follows the final-model prediction file. NRI is the event mean sign of (final minus full) minus the non-event mean sign. IDI is the event mean probability change minus the non-event mean change. Ties contribute zero. The point estimate uses the original sample; 1,000 paired patient-level bootstrap draws use NumPy `default_rng(42)`. Draws without both outcome groups are skipped and the retained count is reported. Intervals are 2.5th–97.5th percentiles; standard errors use sample SD; two-sided p values use a normal approximation (estimate/bootstrap SE). These are exploratory, unadjusted comparisons, not equivalence tests.

Expected rounded results: NRI −0.173 (95% CI −0.738 to 0.400; p=0.557); IDI −0.007 (−0.079 to 0.068; p=0.845). Full model: depth 3, learning rate 0.1, 300 iterations. Final reduced model: depth 3, learning rate 0.03, 300 iterations. Ablation used 600 iterations and is not the source for this comparison. The superseded NRI/IDI code and figure are intentionally excluded.

Outputs include patient-level paired probabilities and must remain in controlled storage. `.gitignore` is a convenience, not a safeguard for files uploaded through the browser.

## Limits of exact reproduction

- Patient data and original fixed assignments are restricted and absent from this repository. A fresh clone alone cannot regenerate the study numbers.
- SHAP was calculated on DEV and describes fitted model behavior; it is not an independent test of explanation stability.
- The uncalibrated threshold uses DEV out-of-fold predictions. Calibrated thresholds use fitted DEV ensemble probabilities, not strictly out-of-fold probabilities; the code preserves this historical procedure and its limitation.
- Model selection uses mean fold average precision; pooled out-of-fold average precision is a different summary.
- Software/OS changes, parallel execution, and floating-point differences can affect reruns.
- Packaging checks covered source syntax, schemas, privacy of included files, synthetic NRI/IDI tests, and reclassification against saved local predictions. The complete model grid was not rerun for this release.
