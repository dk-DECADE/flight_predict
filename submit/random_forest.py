from pathlib import Path
from time import perf_counter
import argparse

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def make_preprocessing(category_columns):

    return ColumnTransformer(
        [("names", OneHotEncoder(handle_unknown="ignore"), category_columns)],
        remainder="passthrough",
    )

def source_features_for_encoded_columns(encoder, category_columns, number_columns):

    source_features = []
    for feature, categories in zip(category_columns, encoder.categories_):
        source_features.extend([feature] * len(categories))
    return source_features + number_columns

def report_feature_importance(output, log, source_features, scores, method):

    importance = pd.Series(scores, index=source_features, dtype=float).groupby(level=0, sort=False).sum()
    total = importance.sum()
    if total > 0:
        importance = importance / total
    ranked = importance.sort_values(ascending=False, kind="stable")
    table = ranked.rename_axis("feature").reset_index(name="importance_share")
    table["method"] = method
    table.to_csv(output / "feature_importance.csv", index=False)
    log(f"\n特征重要度（{method}，按占比从高到低）：")
    for rank, (feature, share) in enumerate(ranked.items(), start=1):
        log(f"  {rank}. {feature:<26} {share:6.2%}")

def evaluate_predictions(actual, probability):
    actual = np.asarray(actual)

    order = np.random.default_rng(42).permutation(len(probability))
    order = order[np.argsort(-probability[order], kind="stable")]
    top_n = int(np.ceil(len(actual) * 0.10))

    selected = order[:top_n]
    hits = int(actual[selected].sum())

    metrics = {"PR_AUC_AP": average_precision_score(actual, probability),
               "Brier": brier_score_loss(actual, probability),
               "Top10名单数量": top_n, "Top10找到的延误数": hits,
               "Precision_Top10": hits / top_n,
               "Recall_Top10": hits / actual.sum()}
    return metrics, selected

def main():
    parser = argparse.ArgumentParser(description="随机森林参数搜索：默认 20 组，可选 100 组。")
    parser.add_argument("--trials", type=int, choices=[20, 100], default=20)
    args = parser.parse_args()

    project_folder = Path(__file__).resolve().parents[1]
    data_folder = project_folder / "清洗完成的数据"
    output_folder = project_folder / "results" / "random_forest"
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / "training_log.txt"
    log_path.write_text("", encoding="utf-8")

    def print_log(message):

        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")

    category_columns = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

    number_columns = [
        "month",
        "day_of_week",
        "scheduled_departure_hour",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
    ]
    feature_columns = category_columns + number_columns

    target = "major_delay_30"

    read_columns = feature_columns + ["FL_DATE", target, "ARR_DELAY", "CANCELLED", "DIVERTED"]

    files = sorted(data_folder.glob("*_cleaned.csv"))
    if not files:
        raise ValueError("没有找到清洗文件，请检查“清洗完成的数据”文件夹。")

    tables = []
    for file in files:
        table = pd.read_csv(file, usecols=read_columns)

        table["record_id"] = file.name + ":" + (table.index + 2).astype(str)
        tables.append(table)
    data = pd.concat(tables, ignore_index=True)
    data["FL_DATE"] = pd.to_datetime(data["FL_DATE"], format="%Y-%m-%d", errors="raise")

    if data[read_columns].isna().any().any():
        raise ValueError("需要用到的字段存在空值，请先检查清洗数据。")
    if data["FL_DATE"].dt.year.nunique() != 1:
        raise ValueError("这一版要求使用同一年的数据。")
    if set(data["FL_DATE"].dt.month) != set(range(1, 13)):
        raise ValueError("这一版使用完整年度，请先准备好 1—12 月数据。")
    if not data["month"].eq(data["FL_DATE"].dt.month).all():
        raise ValueError("month 与实际日期不一致，请检查清洗代码。")
    if not data["day_of_week"].eq(data["FL_DATE"].dt.dayofweek).all():
        raise ValueError("day_of_week 应使用周一=0、周日=6。")
    if not data[target].eq(data["ARR_DELAY"].ge(30).astype(int)).all():
        raise ValueError("延误标签错误：major_delay_30 应等于 ARR_DELAY >= 30。")
    if not data["CANCELLED"].eq(0).all() or not data["DIVERTED"].eq(0).all():
        raise ValueError("这版只使用清洗后未取消、未备降的航班。")
    if not np.isfinite(data[number_columns].to_numpy(dtype=float)).all():
        raise ValueError("数值字段存在无效数字，请检查数据。")

    month = data["FL_DATE"].dt.month

    search_data = data.loc[month.between(1, 6)].copy()
    validation_data = data.loc[month.between(7, 8)].copy()
    train_data = data.loc[month.between(1, 9)].copy()
    test_data = data.loc[month.between(10, 12)].copy()

    split_rows = []
    for name, part in [("调参训练集", search_data), ("验证集", validation_data),
                       ("最终训练集", train_data), ("测试集", test_data)]:
        if part[target].nunique() != 2:
            raise ValueError(name + "必须同时包含延误和未延误的航班。")

        split_rows.append({"数据分组": name, "航班数": len(part),
                           "延误比例": part[target].mean(),
                           "开始日期": part.FL_DATE.min(), "结束日期": part.FL_DATE.max()})
    pd.DataFrame(split_rows).to_csv(output_folder / "data_split.csv", index=False, encoding="utf-8-sig")

    train_data.groupby("month").size().rename("n").to_csv(output_folder / "training_month_counts.csv")
    print_log(f"参数搜索:将比较 {args.trials} 组参数。")
    print_log("1—6 月训练候选，7—8 月验证选参数；1—9 月重训最终模型，10—12 月测试。")
    for row in split_rows:
        print_log(f"{row['数据分组']}：{row['航班数']:,} 趟，延误比例 {row['延误比例']:.2%}，"
                  f"{row['开始日期']:%Y-%m-%d} 至 {row['结束日期']:%Y-%m-%d}")
    print_log("最终训练集各月份记录数：\n" + train_data.groupby("month").size().to_string())

    preprocessing = make_preprocessing(category_columns)
    x_search = preprocessing.fit_transform(search_data[feature_columns])
    x_validation = preprocessing.transform(validation_data[feature_columns])

    tree_options = [50, 100] if args.trials == 20 else [50, 100, 150, 200]
    leaf_options = [20, 50] if args.trials == 20 else [5, 10, 20, 50, 100]

    candidates = [{"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf}
                  for trees in tree_options for depth in [8, 12, 16, 20, 24]
                  for leaf in leaf_options]

    search_rows = []
    best_key = None
    for run, params in enumerate(candidates, start=1):
        print_log(f"\n候选训练 {run}/{len(candidates)}：{params}，随机种子=42")
        forest = RandomForestClassifier(**params, random_state=42, n_jobs=8)
        started = perf_counter()

        forest.fit(x_search, search_data[target])
        fit_seconds = perf_counter() - started
        validation_probability = forest.predict_proba(x_validation)[:, 1]

        metrics, _ = evaluate_predictions(validation_data[target], validation_probability)
        row = {"run": run, **params, "fit_seconds": fit_seconds, **metrics}
        search_rows.append(row)
        pd.DataFrame(search_rows).to_csv(output_folder / "search_results.csv", index=False, encoding="utf-8-sig")

        print_log(f"耗时={fit_seconds:.2f} 秒；验证 AP={metrics['PR_AUC_AP']:.6f}；"
                  f"Brier={metrics['Brier']:.6f}；Precision@Top10%={metrics['Precision_Top10']:.2%}；"
                  f"Recall@Top10%={metrics['Recall_Top10']:.2%}；"
                  f"名单={metrics['Top10名单数量']:,}，找到延误={metrics['Top10找到的延误数']:,}")

        key = (metrics["PR_AUC_AP"], -metrics["Brier"])
        if best_key is None or key > best_key:
            best_key, best_params, best_row = key, params.copy(), row.copy()
        print_log(f"目前最佳：候选 {best_row['run']}，验证 AP={best_row['PR_AUC_AP']:.6f}")

    pd.DataFrame([best_row]).to_csv(output_folder / "best_parameters.csv", index=False, encoding="utf-8-sig")
    print_log(f"\n参数比较完成：选择候选 {best_row['run']}，参数={best_params}")
    print_log("开始最终重训：用全部 1—9 月，包含 9 月；这次不再挑参数。")

    model = Pipeline([("preprocessing", make_preprocessing(category_columns)),
                      ("forest", RandomForestClassifier(**best_params, random_state=42, n_jobs=8))])
    training_started = perf_counter()
    model.fit(train_data[feature_columns], train_data[target])
    training_seconds = perf_counter() - training_started
    print_log(f"最终重训完成：{len(model['forest'].estimators_)} 棵树，耗时 {training_seconds:.2f} 秒。")
    joblib.dump(model, output_folder / "random_forest.joblib", compress=3)

    probability = model.predict_proba(test_data[feature_columns])[:, 1]
    actual = test_data[target].to_numpy()

    metrics, top_positions = evaluate_predictions(actual, probability)

    result = {"数据分组": "测试集", "航班数": len(test_data),
              "真实延误数": int(actual.sum()), "真实延误比例": actual.mean(),
              "平均预测概率": probability.mean(), **metrics}
    pd.DataFrame([result]).to_csv(output_folder / "evaluation.csv", index=False, encoding="utf-8-sig")
    print_log("\n最终模型在 10—12 月的测试结果（不用于挑选模型）：")

    for name, value in result.items():

        print_log(f"  {name} = {value:.6f}" if isinstance(value, (float, np.floating)) else f"  {name} = {value}")

    encoder = model["preprocessing"].named_transformers_["names"]
    source_features = source_features_for_encoded_columns(encoder, category_columns, number_columns)

    report_feature_importance(output_folder, print_log, source_features,
                              model["forest"].feature_importances_, "impurity_decrease")

    prediction_table = test_data[["record_id", "FL_DATE"] + feature_columns + [target]].copy()
    prediction_table["predicted_delay_probability"] = probability
    prediction_table["in_top10_percent"] = 0
    prediction_table.iloc[top_positions, prediction_table.columns.get_loc("in_top10_percent")] = 1
    prediction_table.to_csv(output_folder / "test_predictions.csv.gz", index=False, compression="gzip")
    test_probability = probability
    test_actual = actual

    precision, recall, _ = precision_recall_curve(test_actual, test_probability)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(recall, precision, where="post", label="Random forest")
    ax.axhline(test_actual.mean(), color="gray", linestyle="--", label="Observed delay rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="Random forest: PR curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    observed, predicted = calibration_curve(test_actual, test_probability, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(predicted, observed, "o-", label="Random forest (uncalibrated)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set(xlabel="Predicted probability", ylabel="Observed delay rate",
           title="Random forest: calibration check", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "calibration_curve.png", dpi=150)
    plt.close(fig)

    print_log(f"完成！{len(candidates)} 次候选训练 + 1 次最终重训；结果保存到：{output_folder}")

if __name__ == "__main__":
    main()
