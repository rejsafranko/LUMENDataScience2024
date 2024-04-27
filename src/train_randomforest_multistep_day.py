import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
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


def split_data(df, cutoff_date):
    cutoff_date = pd.to_datetime(cutoff_date)
    train_data = df[df["stay_date"] < cutoff_date].copy()
    test_data = df[df["stay_date"] >= cutoff_date].copy()
    features = [col for col in df.columns if col not in ["stay_date", "room_cnt"]]
    X_train = train_data[features]
    y_train = train_data["room_cnt"]
    X_test = test_data[features]
    y_test = test_data["room_cnt"]
    return train_data, test_data, X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=700, max_depth=6, min_samples_leaf=5, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main(args):
    df = pd.read_parquet(args.dataset_path)
    processed_data = preprocess_data(df)
    processed_data = feature_engineering(processed_data)
    _, test_data, X_train, y_train, _, _ = split_data(processed_data, "2009-1-03")
    model = train_model(X_train, y_train)
    joblib.dump(model, args.model_save_path + "rf_multistep_day.joblib")
    test_data.to_csv("data/evaluation/test_multistep_day.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
