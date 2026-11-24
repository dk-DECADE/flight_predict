# Flight delay prediction · Python and R submission

Both versions predict whether a flight arrives at least 30 minutes late.

For the revised random forest study, start with the [report](random_forest_report.pdf), then run `random_forest_review.py` or `random_forest_review.Rmd`. They run the same three scenarios and also cover the 60-minute target.

| File | Purpose |
|---|---|
| random_forest.py | Random forest with 20 candidate settings by default |
| boosting.py | XGBoost with 40 settings across four boosting sequences |
| flight_delay_models.Rmd | Self-contained R Markdown report for both models |
| random_forest_review.py | Revised random forest protocol for both delay targets and JFK |
| random_forest_review.Rmd | Step-by-step R report for the revised random forest |
| random_forest_report.pdf | Methods, results, diagnostic figures and evaluation limits |
| requirements.txt | Tested Python package versions |

Keep the Python scripts together. XGBoost and the revised random forest reuse evaluation functions from `random_forest.py`.

Place the unpacked `submit/` folder beside `清洗完成的数据/`. The cleaned data directory must contain all 12 monthly `*_cleaned.csv` files from the same calendar year. Install the requirements with Python 3.10 and run these commands from that project directory.

```text
python -m pip install -r submit/requirements.txt
python -X utf8 submit/random_forest.py
python -X utf8 submit/boosting.py
```

In the existing Windows workspace, the interpreter is `E:/anaconda3/envs/pytorch/python.exe`. XGBoost can use the project's `.python-library` install.

Both scripts use the same eight planned flight inputs and the `major_delay_30` label. They fit candidate settings on January-June, select by July-August validation average precision, refit on January-September, then evaluate on October-December. The latter months are held out from fitting and selection.

Random forest compares tree count, tree depth, and minimum leaf size. The option `--trials 100` expands its grid. XGBoost compares depth 3 or 5, learning rate 0.03 or 0.10, and checkpoints at 10, 20, 30, 40, 50, 60, 120, 180, 240, and 300 boosting rounds. Brier score breaks exact ties on validation AP. XGBoost then prefers fewer rounds, followed by the earlier candidate.

Results are saved in `results/random_forest/` and `results/xgboost/`. Each script exports the chosen settings, search scores, test metrics, predictions, training log, saved model, PR curve, and calibration check. XGBoost also saves the per-round validation log loss and its plot. Keep `xgboost.ubj` and `one_hot_encoder.joblib` together for inference.

The alert list contains the highest-risk 10% of flights, rounded up to a whole flight. Selected test flights have `in_top10_percent=1`. The test report includes precision and recall for this list, Average Precision, and Brier score.

Both scripts print a ranked table for the eight original features and save it as `feature_importance.csv`. One-hot columns are summed under their original field. Random forest uses impurity decrease, and XGBoost uses total gain. The shares describe importance within each fitted model.

## Run in RStudio

Open `submit/flight_delay_models.Rmd` in RStudio and click **Knit** to train both models and make the HTML report. Keep the cleaned data folder beside `submit/`. If a package is missing, use the RStudio Console to install it. The XGBoost code requires `xgboost` 3.2 or later.

To run one step at a time, set the RStudio working directory to the project folder or its `submit/` folder. Click **Run Current Chunk** on **Setup**, the shared function blocks, and **Load data**, in that order. For each model, run the blocks to prepare inputs, compare candidates, refit, and score the test set. The candidate blocks show the validation scores and the selected settings. The last section shows the test comparison and where the figures were saved.

```r
packages <- c("rmarkdown", "knitr", "Matrix", "data.table", "ranger", "xgboost", "jsonlite")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing)
```

In the Rmd header, set `model` to `"random_forest"` or `"xgboost"` to run one model. Set `rf_trials` to `100` for the larger random forest grid. The default R outputs go to `results/r/random_forest/` and `results/r/xgboost/`. The HTML includes the test metrics, evaluation plots, validation scores, and feature importance. The output directories also contain per-flight predictions, logs, and fitted models.

Python and R may produce different scores because the random forest engines and random tie-breaking implementations differ.

## Revised random forest

Run `python -X utf8 submit/random_forest_review.py` from the project directory, or open `submit/random_forest_review.Rmd` in RStudio and run its chunks in order. Results go to `results/rf_revision/` for Python and `results/r/rf_revision/` for R. The exploratory RF and XGBoost commands above use the earlier experiment.

In the revised Rmd, set `scenario` to `"delay_30"`, `"delay_60"`, or `"jfk_30"` to run one experiment. Start with `comparison.csv` for the baseline and RF scores, `thresholds.csv` for the September cutoffs and actual test alert rates, and `splits.csv` for the date and JFK checks. Each scenario also has its validation scores, model, predictions, and plots.

The revised protocol fits each forest on January–June, selects by July–August validation AP, and uses September for isotonic probability calibration and a fixed cutoff near a 10% September alert budget. It evaluates 30-minute and 60-minute delays and a separate JFK 30-minute cold start. The cold-start run excludes flights touching JFK from development and all history sources and evaluates October–December flights touching JFK. October–December was inspected during earlier analysis and is a reused evaluation period.

The validation grid compares distance and scheduled duration separately, with and without airport and route history. Weekday and departure hour use categorical encoding. Destinations with fewer than 5,000 training flights are grouped into OTHER. The historical rate cutoff defaults to the previous calendar day. Use `--history-lag 2` in Python or `history_lag: 2` in the Rmd header to match the final report's two-day cutoff. The comparison tables include constant risk, training route risk, rolling route risk, and raw and calibrated RF. The fixed cutoff and exact Top10% are reported separately.

The archive contains the scripts, report, guide, and Python requirements. The repository provides summary tables, validation results, and diagnostic figures for the revised runs. Rerunning the scripts with the cleaned monthly data generates the per-flight predictions and fitted models.
