import argparse
import json
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import OneHotEncoder

from random_forest import evaluate_predictions, plt, report_feature_importance, source_features_for_encoded_columns

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
NUMBERS = ["month", "day_of_week", "scheduled_departure_hour", "CRS_ELAPSED_TIME", "DISTANCE"]
FEATURES = CATEGORIES + NUMBERS
TARGET = "major_delay_30"
SEED = 1012508
MAX_BIN = 64
THREADS = 8

def read_flights(folder):

    columns = FEATURES + ["FL_DATE", TARGET, "ARR_DELAY", "CANCELLED", "DIVERTED"]
    files = sorted(folder.glob("*_cleaned.csv"))
    if not files:
        raise ValueError(f"找不到清洗后的航班数据：{folder}")
    frames = []
    for file in files:
        frame = pd.read_csv(file, usecols=columns)
        frame["record_id"] = file.name + ":" + (frame.index + 2).astype(str)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(data["FL_DATE"], format="%Y-%m-%d", errors="raise")

    if data[columns].isna().any().any() or not np.isfinite(data[NUMBERS].to_numpy(dtype=float)).all():
        raise ValueError("航班数据有空值或非有限数值。")
    if dates.dt.year.nunique() != 1 or set(dates.dt.month) != set(range(1, 13)):
        raise ValueError("需要同一年的 1—12 月数据。")
    if not data.month.eq(dates.dt.month).all() or not data.day_of_week.eq(dates.dt.dayofweek).all():
        raise ValueError("月份或星期字段与日期不一致。")
    if not data[TARGET].eq(data.ARR_DELAY.ge(30).astype(int)).all():
        raise ValueError("延误标签应等于 ARR_DELAY >= 30。")
    if not data.CANCELLED.eq(0).all() or not data.DIVERTED.eq(0).all():
        raise ValueError("请使用已排除取消和备降航班的清洗数据。")
    return data.drop(columns=["ARR_DELAY", "CANCELLED", "DIVERTED"])

def encode_features(flights, encoder):

    indicators = encoder.transform(flights[CATEGORIES])
    values = flights[NUMBERS].to_numpy(dtype=np.float32)
    rows, columns = values.shape

    numeric = csr_matrix((values.ravel(), np.tile(np.arange(columns), rows),
                          np.arange(rows + 1) * columns), shape=values.shape)
    return hstack([indicators, numeric], format="csr")

def plot_evaluation(output, actual, probability, best_round, history):
    precision, recall, _ = precision_recall_curve(actual, probability)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where="post", label="XGBoost")
    ax.axhline(actual.mean(), color="gray", linestyle="--", label="Observed delay rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="Oct-Dec test: precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    observed, predicted = calibration_curve(actual, probability, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(predicted, observed, "o-", label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted risk", ylabel="Observed delay rate",
           title="Oct-Dec test: raw probability check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (depth, rate), trace in history.groupby(["max_depth", "learning_rate"]):
        ax.plot(trace.boosting_round, trace.validation_logloss, label=f"Depth {depth}, learning rate {rate:g}")
    ax.axvline(best_round, color="gray", linestyle="--", label="Round chosen by validation AP")
    ax.set(xlabel="Boosting round", ylabel="Validation log loss", title="July-August validation: loss by round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "boosting_learning_curve.png", dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="比较多组梯度提升树设置，并用 1—9 月重训。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "清洗完成的数据")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/xgboost")
    args = parser.parse_args()
    local_library = ROOT / ".python-library"
    if local_library.is_dir():
        sys.path.insert(0, str(local_library))
    import xgboost as xgb

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    log_file = output / "training_log.txt"
    log_file.write_text("", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    data = read_flights(args.data_dir)
    search_train = data.loc[data.month.between(1, 6)]
    validation = data.loc[data.month.between(7, 8)]
    final_train = data.loc[data.month.between(1, 9)]
    test = data.loc[data.month.between(10, 12)]
    splits = []
    log("1—6 月训练候选，7—8 月验证，1—9 月最终重训，10—12 月测试。")
    for role, frame in [("search_train", search_train), ("validation", validation),
                        ("final_train", final_train), ("test", test)]:
        if frame[TARGET].nunique() != 2:
            raise ValueError(f"{role} 需要同时有延误与未延误航班。")

        splits.append({"split": role, "n": len(frame), "delay_rate": frame[TARGET].mean(),
                       "first_date": frame.FL_DATE.min(), "last_date": frame.FL_DATE.max()})
        log(f"{role}: {len(frame):,} 趟，{frame.FL_DATE.min()} 至 {frame.FL_DATE.max()}")
    pd.DataFrame(splits).to_csv(output / "data_split.csv", index=False)
    final_train.groupby("month").size().rename("n").to_csv(output / "training_month_counts.csv")

    search_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32).fit(search_train[CATEGORIES])

    train_matrix = xgb.QuantileDMatrix(encode_features(search_train, search_encoder),
                                      label=search_train[TARGET], max_bin=MAX_BIN, nthread=THREADS)

    validation_matrix = xgb.QuantileDMatrix(encode_features(validation, search_encoder),
                                           label=validation[TARGET], ref=train_matrix,
                                           max_bin=MAX_BIN, nthread=THREADS)

    settings = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_bin": MAX_BIN,
        "nthread": THREADS,
        "seed": SEED,
    }

    rounds = [10, 20, 30, 40, 50, 60, 120, 180, 240, 300]

    sequences = list(product([3, 5], [0.03, 0.10]))
    total_candidates = len(sequences) * len(rounds)
    candidates = []
    histories = []
    log(f"比较 {len(sequences)} 组深度和学习率搭配，每组检查 {len(rounds)} 个轮数，共 {total_candidates} 组设置。")

    for depth, rate in sequences:

        run_history = {}

        log(f"开始训练：深度={depth}，学习率={rate:g}，共 {rounds[-1]} 轮。")
        started = perf_counter()
        model = xgb.train(
            dict(settings, max_depth=depth, eta=rate),
            train_matrix,
            num_boost_round=rounds[-1],
            evals=[(validation_matrix, "validation")],
            evals_result=run_history,
            verbose_eval=20,
        )
        seconds = perf_counter() - started

        log(f"这一条 {rounds[-1]} 轮序列用时 {seconds:.2f} 秒。")

        histories.extend(
            {"max_depth": depth, "learning_rate": rate,
             "boosting_round": r, "validation_logloss": loss}

            for r, loss in enumerate(run_history["validation"]["logloss"], start=1)
        )
        for checkpoint in rounds:

            probability = model.predict(validation_matrix, iteration_range=(0, checkpoint))

            metrics, _ = evaluate_predictions(validation[TARGET], probability)
            row = {
                "run": len(candidates) + 1,
                "max_depth": depth,
                "learning_rate": rate,
                "boosting_rounds": checkpoint,

                "validation_logloss": run_history["validation"]["logloss"][checkpoint - 1],
                **metrics,
            }
            candidates.append(row)

            log(f"候选 {row['run']}/{total_candidates} | 深度={depth} 学习率={rate:g} 轮数={checkpoint} | "
                f"验证 AP={metrics['PR_AUC_AP']:.6f} Brier={metrics['Brier']:.6f} "
                f"Top10% 命中率={metrics['Precision_Top10']:.2%} 覆盖率={metrics['Recall_Top10']:.2%}")

        pd.DataFrame(candidates).to_csv(output / "search_results.csv", index=False, encoding="utf-8-sig")

        pd.DataFrame(histories).to_csv(output / "round_history.csv", index=False)

    search = pd.DataFrame(candidates)

    best = search.sort_values(["PR_AUC_AP", "Brier", "boosting_rounds", "run"],
                              ascending=[False, True, True, True]).iloc[0]

    search.loc[search.run.eq(best.run)].to_csv(output / "best_parameters.csv", index=False, encoding="utf-8-sig")
    best_round = int(best.boosting_rounds)

    final_settings = dict(settings, max_depth=int(best.max_depth), eta=float(best.learning_rate))
    del train_matrix, validation_matrix, model

    log(f"选中候选 {int(best.run)}：深度={int(best.max_depth)}，学习率={best.learning_rate:g}，"
        f"轮数={best_round}，验证 AP={best.PR_AUC_AP:.6f}。开始用包含 9 月的完整训练集重训。")
    final_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32).fit(final_train[CATEGORIES])

    final_matrix = xgb.QuantileDMatrix(encode_features(final_train, final_encoder),
                                      label=final_train[TARGET], max_bin=MAX_BIN, nthread=THREADS)
    model = xgb.train(final_settings, final_matrix, num_boost_round=best_round)
    model.save_model(output / "xgboost.ubj")
    joblib.dump(final_encoder, output / "one_hot_encoder.joblib")

    metadata = {"features": FEATURES, "target": TARGET, "training_rows": len(final_train),
                "boosting_rounds": best_round, "parameters": final_settings, "xgboost_version": xgb.__version__}
    (output / "model_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    probability = model.predict(xgb.DMatrix(encode_features(test, final_encoder), nthread=THREADS))
    actual = test[TARGET].to_numpy()
    metrics, selected = evaluate_predictions(actual, probability)

    result = {"数据分组": "测试集", "航班数": len(test), "真实延误数": int(actual.sum()),
              "真实延误比例": actual.mean(), "平均预测概率": probability.mean(), **metrics}
    pd.DataFrame([result]).to_csv(output / "evaluation.csv", index=False, encoding="utf-8-sig")
    exported = test[["record_id", "FL_DATE"] + FEATURES + [TARGET]].copy()
    exported["predicted_delay_probability"] = probability
    exported["in_top10_percent"] = 0
    exported.iloc[selected, exported.columns.get_loc("in_top10_percent")] = 1
    exported.to_csv(output / "test_predictions.csv.gz", index=False, compression="gzip")
    plot_evaluation(output, actual, probability, best_round, pd.DataFrame(histories))

    log(f"最终测试 | AP={metrics['PR_AUC_AP']:.6f} Brier={metrics['Brier']:.6f} "
        f"Top10% 命中率={metrics['Precision_Top10']:.2%} 覆盖率={metrics['Recall_Top10']:.2%}")

    source_features = source_features_for_encoded_columns(final_encoder, CATEGORIES, NUMBERS)
    gain = model.get_score(importance_type="total_gain")
    scores = [gain.get(f"f{index}", 0.0) for index in range(model.num_features())]
    report_feature_importance(output, log, source_features, scores, "total_gain")
    log(f"完成。结果已保存到 {output}")

if __name__ == "__main__":
    main()
