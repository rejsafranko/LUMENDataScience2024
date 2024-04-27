import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X has feature names, but DecisionTreeRegressor was fitted without feature names",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'M' is deprecated and will be removed in a future version, please use 'ME' instead.",
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()


def get_expand_counts(series, prefix):
    counts = series.value_counts().to_dict()
    return pd.DataFrame({f"{prefix}_{k}": [v] for k, v in counts.items()})


def calculate_aggregates(df):
    agg_funcs = {
        "night_number": pd.NamedAgg(column="night_number", aggfunc="mean"),
        "lead_time_days": pd.NamedAgg(column="lead_time_days", aggfunc="mean"),
        "adult_cnt": pd.NamedAgg(column="adult_cnt", aggfunc="mean"),
        "total_price": pd.NamedAgg(column="total_price", aggfunc="mean"),
        "room_cnt": pd.NamedAgg(
            column="room_cnt",
            aggfunc=lambda x: x[
                df.loc[x.index, "reservation_status"] == "Checked-out"
            ].sum(),
        ),
        "guest_country_id_HR_count": pd.NamedAgg(
            column="guest_country_id", aggfunc=lambda x: (x == "HR").sum()
        ),
        "guest_country_id_not_HR_count": pd.NamedAgg(
            column="guest_country_id", aggfunc=lambda x: (x != "HR").sum()
        ),
    }
    grouped = df.groupby("stay_date").agg(**agg_funcs).reset_index()
    return grouped.fillna(0)


def calculate_expected_room_count(df):
    conditions = (
        (df["reservation_date"] < (df["stay_date"] - pd.Timedelta(days=29)))
        & (df["date_from"] <= df["stay_date"])
        & (df["date_to"] >= df["stay_date"])
        & (~df["reservation_status"].isin(["Cancelled", "No-show"]))
    )
    valid_reservations = df[conditions]
    return (
        valid_reservations.groupby("stay_date")
        .agg(expected_room_cnt=("room_cnt", "sum"))
        .reset_index()
    )


def merge_dataframes(df1, df2):
    return df1.merge(df2, on="stay_date", how="left").fillna(0)


def resample_data(df):
    df_resampled = df.set_index("stay_date").resample("30D").sum().reset_index()
    df_resampled[["room_cnt", "expected_room_cnt"]] = df_resampled[
        ["room_cnt", "expected_room_cnt"]
    ].shift(1)
    df_resampled["stay_date"] = df_resampled["stay_date"].shift(1)
    return df_resampled.dropna()


def add_cyclical_features(df, col_name, max_val):
    df[col_name + "_sin"] = np.sin(2 * np.pi * df[col_name] / max_val)
    df[col_name + "_cos"] = np.cos(2 * np.pi * df[col_name] / max_val)
    return df


def feature_engineering(df):
    df["month"] = df["stay_date"].dt.month
    df["quarter"] = df["stay_date"].dt.quarter
    df = add_cyclical_features(df, "month", 12)
    return df


def get_data_for_prediction(df):
    features = [col for col in df.columns if col not in ["stay_date", "room_cnt"]]
    X = df[features]
    y = df["room_cnt"]
    return df, X, y


def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(len(features)), importances[indices], color="r", align="center")
    plt.xticks(range(len(features)), np.array(features)[indices], rotation=90)
    plt.xlim([-1, len(features)])
    plt.show()


def predict_intervals(model, X_test, percentile=95):
    predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
    lower_bound = np.percentile(predictions, (100 - percentile) / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2, axis=0)
    return lower_bound, upper_bound


def plot_results(X_test_dates, y_test, y_pred, lower_bound, upper_bound):
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_dates, y_test, label="Actual", color="blue")
    plt.plot(X_test_dates, y_pred, label="Predicted", color="red")
    plt.fill_between(
        X_test_dates,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.5,
        label="Prediction Interval",
    )
    plt.title("Room Count Forecast with Prediction Intervals")
    plt.xlabel("Date")
    plt.ylabel("Room Count")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_precision(y_test, lower_bound, upper_bound):
    within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean()
    print(
        f"Percentage of actual values within predicted interval: {within_interval * 100:.2f}%"
    )


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")


def main(args):
    df = pd.read_parquet("data/processed/train.parquet")
    df["reservation_date"] = pd.to_datetime(df["reservation_date"])
    df["date_from"] = pd.to_datetime(df["date_from"])
    df["date_to"] = pd.to_datetime(df["date_to"])
    df["lead_time_days"] = (df["date_from"] - df["reservation_date"]).dt.days
    checked_out_reservations = df[df["reservation_status"] == "Checked-out"]
    df = checked_out_reservations.copy()
    df = df[(df["stay_date"] >= df["date_from"]) & (df["stay_date"] <= df["date_to"])]
    df = df[df["date_from"] <= df["date_to"]]
    df = df[df["reservation_date"] <= df["date_from"]]

    # Calculate aggregates and expected room count.
    grouped_df = calculate_aggregates(df.copy())
    expected_room_cnt = calculate_expected_room_count(df)
    final_result = merge_dataframes(grouped_df, expected_room_cnt)
    processed_data = feature_engineering(resample_data(final_result.copy()))

    model = joblib.load(args.model_path)
    processed_data = feature_engineering(processed_data)
    test_data, X_test, y_test = get_data_for_prediction(processed_data)
    y_pred = model.predict(X_test)
    lower_bound, upper_bound = predict_intervals(model, X_test)
    plot_results(test_data["stay_date"], y_test, y_pred, lower_bound, upper_bound)
    evaluate_precision(y_test, lower_bound, upper_bound)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)
