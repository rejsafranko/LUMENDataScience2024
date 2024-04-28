import numpy as np
import pandas as pd

def add_cyclical_features(df, col_name, max_val):
    df[col_name + '_sin'] = np.sin(2 * np.pi * df[col_name] / max_val)
    df[col_name + '_cos'] = np.cos(2 * np.pi * df[col_name] / max_val)
    return df

def feature_engineering_day(df):
    df['year'] = df['stay_date'].dt.year - 2007
    df['month'] = df['stay_date'].dt.month
    df['day'] = df['stay_date'].dt.day
    df['week_of_year'] = df['stay_date'].dt.isocalendar().week
    df['quarter'] = df['stay_date'].dt.quarter
    df['day_of_week'] = df['stay_date'].dt.dayofweek
    df['day_of_year'] = df['stay_date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df = add_cyclical_features(df, 'day', 31)
    df = add_cyclical_features(df, 'week_of_year', 52)
    df = add_cyclical_features(df, 'month', 12)
    df = add_cyclical_features(df, 'day_of_week', 7)
    df = add_cyclical_features(df, 'day_of_year', 365)
    return df

def feature_engineering_week(df):
    df['year'] = df['stay_date'].dt.year - 2007
    df['month'] = df['stay_date'].dt.month
    df['week_of_year'] = df['stay_date'].dt.isocalendar().week
    df['quarter'] = df['stay_date'].dt.quarter

    df = add_cyclical_features(df, 'week_of_year', 52)
    df = add_cyclical_features(df, 'month', 12)
    return df

def feature_engineering_month(df):
    df['year'] = df['stay_date'].dt.year - 2007
    df['month'] = df['stay_date'].dt.month
    df['quarter'] = df['stay_date'].dt.quarter

    df = add_cyclical_features(df, 'month', 12)
    return df

def resample_data7D(df):
    df_resampled = df.set_index('stay_date').resample('7D').sum().reset_index()
    df_resampled[['room_cnt', 'expected_room_cnt']] = df_resampled[['room_cnt', 'expected_room_cnt']].shift(1)
    df_resampled['stay_date'] = df_resampled['stay_date'].shift(1)
    return df_resampled.dropna()

def resample_data30D(df):
    df_resampled = df.set_index('stay_date').resample('30D').sum().reset_index()
    df_resampled[['room_cnt', 'expected_room_cnt']] = df_resampled[['room_cnt', 'expected_room_cnt']].shift(1)
    df_resampled['stay_date'] = df_resampled['stay_date'].shift(1)
    return df_resampled.dropna()

def calculate_aggregates(df):
    agg_funcs = {
        'night_number': pd.NamedAgg(column='night_number', aggfunc='mean'),
        'lead_time_days': pd.NamedAgg(column='lead_time_days', aggfunc='mean'),
        'adult_cnt': pd.NamedAgg(column='adult_cnt', aggfunc='mean'),
        'total_price': pd.NamedAgg(column='total_price', aggfunc='mean'),
        'room_cnt': pd.NamedAgg(column='room_cnt', aggfunc=lambda x: x[df.loc[x.index, 'reservation_status'] == 'Checked-out'].sum()),
        'guest_country_id_HR_count': pd.NamedAgg(column='guest_country_id', aggfunc=lambda x: (x == 'HR').sum()),
        'guest_country_id_not_HR_count': pd.NamedAgg(column='guest_country_id', aggfunc=lambda x: (x != 'HR').sum())
    }
    grouped = df.groupby('stay_date').agg(**agg_funcs).reset_index()
    return grouped.fillna(0)

def calculate_expected_room_count7D(df):
    conditions = (
        (df['reservation_date'] < (df['stay_date'] - pd.Timedelta(days=6))) &
        (df['date_from'] <= df['stay_date']) &
        (df['date_to'] >= df['stay_date']) &
        (~df['reservation_status'].isin(['Cancelled', 'No-show']))
    )
    valid_reservations = df[conditions]
    return valid_reservations.groupby('stay_date').agg(expected_room_cnt=('room_cnt', 'sum')).reset_index()

def calculate_expected_room_count30D(df):
    conditions = (
        (df['reservation_date'] < (df['stay_date'] - pd.Timedelta(days=29))) &
        (df['date_from'] <= df['stay_date']) &
        (df['date_to'] >= df['stay_date']) &
        (~df['reservation_status'].isin(['Cancelled', 'No-show']))
    )
    valid_reservations = df[conditions]
    return valid_reservations.groupby('stay_date').agg(expected_room_cnt=('room_cnt', 'sum')).reset_index()


def merge_dataframes(df1, df2):
    return df1.merge(df2, on='stay_date', how='left').fillna(0)