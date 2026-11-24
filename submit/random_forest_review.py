import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from random_forest import evaluate_predictions, plt

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "day_of_week", "scheduled_departure_hour"]
HISTORICAL = ["origin_rate", "route_rate"]
TRIP_FEATURES = ["DISTANCE", "CRS_ELAPSED_TIME"]
TARGETS = ["major_delay_30", "major_delay_60"]
SCENARIOS = [("delay_30", TARGETS[0], False), ("delay_60", TARGETS[1], False),
             ("jfk_30", TARGETS[0], True)]

def read_flights(folder):

    files = sorted(folder.glob("*_cleaned.csv"))
    if not files:
        raise ValueError(f"找不到清洗后的月度 CSV：{folder}")
    columns = CATEGORIES + TRIP_FEATURES + TARGETS + ["month", "FL_DATE", "ARR_DELAY", "CANCELLED", "DIVERTED"]
    parts = []
    for path in files:
        month = pd.read_csv(path, usecols=columns)
        month["record_id"] = path.name + ":" + (month.index + 2).astype(str)
        parts.append(month)
    flights = pd.concat(parts, ignore_index=True)
    flights["day"] = pd.to_datetime(flights["FL_DATE"], format="%Y-%m-%d")
    month = flights["day"].dt.month
    if flights[columns].isna().any().any() or not np.isfinite(flights[TRIP_FEATURES + ["ARR_DELAY"]]).all().all():
        raise ValueError("清洗数据有缺失或无效数值。")
    if flights["day"].dt.year.nunique() != 1 or set(month) != set(range(1, 13)):
        raise ValueError("请使用同一年度的 12 个月数据。")
    if not (flights["month"].eq(month) & flights["day_of_week"].eq(flights["day"].dt.dayofweek)).all():
        raise ValueError("月份或星期与 FL_DATE 不一致。")
    if not (flights["CANCELLED"].eq(0) & flights["DIVERTED"].eq(0)).all():
        raise ValueError("数据应已排除取消和备降航班。")
    for minutes, target in zip([30, 60], TARGETS):
        if not flights[target].eq(flights["ARR_DELAY"].ge(minutes).astype(int)).all():
            raise ValueError(f"{target} 与实际到达延误不一致。")
    return flights.drop(columns=["ARR_DELAY", "CANCELLED", "DIVERTED"])

def daily_table(reference, calendar, keys, target):

    counts = reference.groupby(keys + ["day"])[target].agg(n="size", hits="sum").reset_index()
    grid = reference[keys].drop_duplicates().merge(pd.DataFrame({"day": calendar}), how="cross")
    table = grid.merge(counts, how="left", on=keys + ["day"]).sort_values(keys + ["day"])
    table[["n", "hits"]] = table[["n", "hits"]].fillna(0)
    return table

def historical_rates(flights, reference, target, lag):

    calendar = pd.date_range(flights["day"].min(), flights["day"].max())
    global_counts = reference.groupby("day")[target].agg(n="size", hits="sum").reindex(calendar, fill_value=0)
    global_past = global_counts.cumsum().shift(lag, fill_value=0)
    global_rate = global_past["hits"].div(global_past["n"].replace(0, np.nan))
    global_rate = flights["day"].map(global_rate).fillna(0)

    airport = daily_table(reference, calendar, ["ORIGIN"], target)
    airport_past = airport.groupby("ORIGIN")[["n", "hits"]].shift(lag)
    airport["rate"] = airport_past["hits"].div(airport_past["n"].replace(0, np.nan))
    origin_keys = pd.MultiIndex.from_frame(flights[["ORIGIN", "day"]])
    origin = airport.set_index(["ORIGIN", "day"])["rate"].reindex(origin_keys).to_numpy()
    origin = pd.Series(origin, index=flights.index).fillna(global_rate)

    route_keys = ["ORIGIN", "DEST"]
    routes = daily_table(reference, calendar, route_keys, target)
    totals = routes.groupby(route_keys)[["n", "hits"]].cumsum()
    by_route = totals.groupby([routes[key] for key in route_keys])
    past_28 = by_route.shift(lag, fill_value=0) - by_route.shift(lag + 28, fill_value=0)

    routes["rate"] = past_28["hits"].div(past_28["n"].where(past_28["n"].ge(20)))
    route_index = pd.MultiIndex.from_frame(flights[route_keys + ["day"]])
    route = routes.set_index(route_keys + ["day"])["rate"].reindex(route_index).to_numpy()
    return pd.DataFrame({"origin_rate": origin, "route_rate": pd.Series(route, index=flights.index).fillna(origin)})

def static_route_rate(training, test, target):

    counts = training.groupby(["ORIGIN", "DEST"])[target].agg(n="size", risk="mean")
    rates = counts["risk"].where(counts["n"].ge(20))
    keys = pd.MultiIndex.from_frame(test[["ORIGIN", "DEST"]])
    route = pd.Series(rates.reindex(keys).to_numpy(), index=test.index)
    airport = test["ORIGIN"].map(training.groupby("ORIGIN")[target].mean())
    return route.fillna(airport).fillna(training[target].mean()).to_numpy()

def training_encoder(training, numeric):

    counts = training["DEST"].value_counts()
    destinations = counts[counts.ge(5000)].index.to_list()
    encoder = ColumnTransformer(
        [("category", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATEGORIES)],
        remainder="passthrough", sparse_threshold=1.0,
    )
    encoder.fit(model_input(training, numeric, destinations))
    return encoder, destinations

def model_input(flights, numeric, destinations):

    inputs = flights[CATEGORIES + numeric].copy()
    inputs["DEST"] = inputs["DEST"].where(inputs["DEST"].isin(destinations), "OTHER")
    return inputs

def score(actual, probability):
    metrics, top = evaluate_predictions(actual, probability)
    return {
        "AP": metrics["PR_AUC_AP"], "ROC_AUC": roc_auc_score(actual, probability),
        "Brier": metrics["Brier"], "Top10_precision": metrics["Precision_Top10"],
        "Top10_recall": metrics["Recall_Top10"], "Top10_count": len(top),
        "Top10_hits": int(actual[top].sum()),
    }, top

def model_plots(output, actual, methods):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, probability in methods.items():
        precision, recall, _ = precision_recall_curve(actual, probability)
        ax.step(recall, precision, where="post", label=method)
    ax.set(xlabel="Recall", ylabel="Precision", title="October-December precision–recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for method in ["RF raw", "RF calibrated"]:
        observed, predicted = calibration_curve(actual, methods[method], n_bins=10)
        ax.plot(predicted, observed, "o-", label=method)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed delay rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)

def run_experiment(flights, name, target, holdout, output, lag, trees, jobs):
    output.mkdir(parents=True, exist_ok=True)
    touches_jfk = flights["ORIGIN"].eq("JFK") | flights["DEST"].eq("JFK")
    if holdout:

        reference = flights.loc[~touches_jfk]
        eligible = (~touches_jfk & flights["month"].le(9)) | (touches_jfk & flights["month"].ge(10))
        flights = flights.loc[eligible].copy()
    else:
        reference = flights
        flights = flights.copy()
    flights[HISTORICAL] = historical_rates(flights, reference, target, lag)
    del reference
    train = flights.loc[flights["month"].le(6)]
    validation = flights.loc[flights["month"].between(7, 8)]
    september = flights.loc[flights["month"].eq(9)]
    test = flights.loc[flights["month"].ge(10)]
    split_rows = []
    for role, part in [("train", train), ("validation", validation), ("calibration", september), ("test", test)]:
        if part[target].nunique() != 2:
            raise ValueError(f"{name} 的 {role} 必须包含延误和未延误两类。")
        split_rows.append({"scenario": name, "split": role, "flights": len(part),
                           "delays": int(part[target].sum()), "jfk_flights": int(touches_jfk.loc[part.index].sum()),
                           "first_date": part["day"].min().date(), "last_date": part["day"].max().date()})
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)

    print(f"\n{name}: 训练 {len(train):,}，验证 {len(validation):,}，校准 {len(september):,}，测试 {len(test):,}", flush=True)
    train_y = train[target].to_numpy()
    validation_y = validation[target].to_numpy()
    rows = []
    best = None

    for trip, history in product(TRIP_FEATURES, [False, True]):
        numeric = ["month", trip] + (HISTORICAL if history else [])
        encoder, destinations = training_encoder(train, numeric)
        train_x = encoder.transform(model_input(train, numeric, destinations))
        validation_x = encoder.transform(model_input(validation, numeric, destinations))
        for depth in [8, 12]:
            model = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=20,
                                           random_state=42, n_jobs=jobs)
            model.fit(train_x, train_y)
            probability = model.predict_proba(validation_x)[:, 1]
            measures, _ = score(validation_y, probability)
            row = {"candidate": len(rows) + 1, "trip_feature": trip, "history": history,
                   "max_depth": depth, "n_estimators": trees, **measures}
            rows.append(row)
            ranking = (-row["AP"], row["Brier"], row["candidate"])
            if best is None or ranking < best[0]:
                best = (ranking, row, model, encoder, destinations, numeric)
            print(f"  {row['candidate']}/8 {trip}, history={history}, depth={depth} | "
                  f"AP={row['AP']:.5f} Brier={row['Brier']:.5f} "
                  f"Top10% 命中率={row['Top10_precision']:.2%} 召回率={row['Top10_recall']:.2%}", flush=True)
            pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)

    _, selected, model, encoder, destinations, numeric = best
    print(f"  选中候选 {selected['candidate']}，验证 AP={selected['AP']:.5f}。用 9 月校准和选阈值。", flush=True)
    pd.DataFrame([selected]).to_csv(output / "best_parameters.csv", index=False)

    sept_x = encoder.transform(model_input(september, numeric, destinations))
    sept_raw = model.predict_proba(sept_x)[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(sept_raw, september[target])
    sept_probability = calibrator.predict(sept_raw)
    threshold = np.sort(sept_probability)[-int(np.ceil(len(september) * 0.10))]
    september_alert_rate = float(np.mean(sept_probability >= threshold))
    joblib.dump({"model": model, "encoder": encoder, "destinations": destinations, "numeric": numeric,
                 "calibrator": calibrator, "threshold": threshold, "target": target, "history_lag": lag},
                output / "model.joblib", compress=3)

    actual = test[target].to_numpy()
    test_x = encoder.transform(model_input(test, numeric, destinations))
    raw = model.predict_proba(test_x)[:, 1]
    calibrated = calibrator.predict(raw)
    methods = {
        "RF raw": raw, "RF calibrated": calibrated,
        "Constant risk": np.full(len(test), train_y.mean()),
        "Training route rate": static_route_rate(train, test, target),
        "Rolling route rate": test["route_rate"].to_numpy(),
    }
    rows = []
    for method, probability in methods.items():
        measures, top = score(actual, probability)
        rows.append({"scenario": name, "target": target, "method": method, "flights": len(test),
                     "actual_delays": int(actual.sum()), **measures})
        if method == "RF calibrated":
            top10 = np.zeros(len(test), dtype=bool)
            top10[top] = True
    fixed = calibrated >= threshold
    fixed_hits = int(actual[fixed].sum())
    threshold_row = {"scenario": name, "target": target, "threshold": threshold,
                     "september_alert_rate": september_alert_rate, "test_alert_rate": fixed.mean(),
                     "test_alerts": int(fixed.sum()), "test_hits": fixed_hits,
                     "test_precision": fixed_hits / fixed.sum() if fixed.any() else np.nan,
                     "test_recall": fixed_hits / actual.sum()}
    result = pd.DataFrame(rows)
    result.to_csv(output / "evaluation.csv", index=False)
    pd.DataFrame([threshold_row]).to_csv(output / "threshold.csv", index=False)
    predictions = test[["record_id", "FL_DATE", "ORIGIN", "DEST", target]].copy()
    predictions["origin_rate"] = test["origin_rate"]
    for method, probability in methods.items():
        predictions[method.lower().replace(" ", "_")] = probability
    predictions["in_top10_percent"] = top10
    predictions["above_fixed_threshold"] = fixed
    predictions.to_csv(output / "test_predictions.csv.gz", index=False)
    model_plots(output, actual, methods)
    print(result.round(5).to_string(index=False), flush=True)
    return result, threshold_row, split_rows

def main():
    parser = argparse.ArgumentParser(description="随机森林整改：双目标、独立校准、历史基准和 JFK 冷启动")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "清洗完成的数据")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rf_revision")
    parser.add_argument("--history-lag", type=int, choices=[1, 2], default=1)
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.trees < 1 or args.jobs < 1:
        parser.error("--trees 和 --jobs 必须大于 0")
    flights = read_flights(args.data_dir)
    comparisons, thresholds, splits = [], [], []
    for name, target, jfk in SCENARIOS:
        result, threshold, split = run_experiment(flights, name, target, jfk, args.output_dir / name,
                                                  args.history_lag, args.trees, args.jobs)
        comparisons.append(result)
        thresholds.append(threshold)
        splits.extend(split)
        pd.concat(comparisons, ignore_index=True).to_csv(args.output_dir / "comparison.csv", index=False)
        pd.DataFrame(thresholds).to_csv(args.output_dir / "thresholds.csv", index=False)
        pd.DataFrame(splits).to_csv(args.output_dir / "splits.csv", index=False)

if __name__ == "__main__":
    main()
