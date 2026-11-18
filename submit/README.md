# Flight delay prediction · Python submission

Two training scripts predict whether a flight arrives at least 30 minutes late.

| File | Purpose |
|---|---|
| random_forest.py | Random forest with 20 candidate settings by default |
| boosting.py | XGBoost with 40 settings across four boosting sequences |
| requirements.txt | Tested Python package versions |

Keep both scripts in the same directory. The Boosting script imports the shared evaluation function from the random forest script.

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

The archive contains the two scripts, this guide, and requirements. The dataset and generated results are supplied separately in the workspace.
