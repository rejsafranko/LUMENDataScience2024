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


def add_cyclical_features(df, col_name, max_val):
    df[col_name + "_sin"] = np.sin(2 * np.pi * df[col_name] / max_val)
    df[col_name + "_cos"] = np.cos(2 * np.pi * df[col_name] / max_val)
    return df


def preprocess_data(df):
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df = df.groupby("stay_date")["room_cnt"].sum().reset_index()
    df = df.set_index("stay_date").resample("D").sum().reset_index()
    return df


def feature_engineering(df):
    df["year"] = df["stay_date"].dt.year - 2007
    df["month"] = df["stay_date"].dt.month
    df["day"] = df["stay_date"].dt.day
    df["week_of_year"] = df["stay_date"].dt.isocalendar().week
    df["quarter"] = df["stay_date"].dt.quarter
    df["day_of_week"] = df["stay_date"].dt.dayofweek
    df["day_of_year"] = df["stay_date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    df = add_cyclical_features(df, "day", 31)
    df = add_cyclical_features(df, "week_of_year", 52)
    df = add_cyclical_features(df, "month", 12)
    df = add_cyclical_features(df, "day_of_week", 7)
    df = add_cyclical_features(df, "day_of_year", 365)
    return df


def get_data_for_prediction(df):
    features = [col for col in df.columns if col not in ["stay_date", "room_cnt"]]
    X = df[features]
    y = df["room_cnt"]
    return df, X, y


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

    model = joblib.load(args.model_path)
    processed_data = preprocess_data(df)
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
