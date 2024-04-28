import pandas as pd

def clean(df):
    df['reservation_date'] = pd.to_datetime(df['reservation_date'])
    df['date_from'] = pd.to_datetime(df['date_from'])
    df['date_to'] = pd.to_datetime(df['date_to'])
    df['lead_time_days'] = (df['date_from'] - df['reservation_date']).dt.days
    checked_out_reservations = df[df['reservation_status'] == 'Checked-out']
    df = checked_out_reservations.copy()
    df = df[(df['stay_date'] >= df['date_from']) & (df['stay_date'] <= df['date_to'])]
    df = df[df['date_from'] <= df['date_to']]
    df = df[df['reservation_date'] <= df['date_from']]
    return df

def preprocess_data_day(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.groupby('stay_date')['room_cnt'].sum().reset_index()
    df = df.set_index('stay_date').resample('D').sum().reset_index()
    return df

def preprocess_data_week(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.groupby('stay_date')['room_cnt'].sum().reset_index()
    df = df.set_index('stay_date').resample('W').sum().reset_index()
    return df

def preprocess_data_month(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.groupby('stay_date')['room_cnt'].sum().reset_index()
    df = df.set_index('stay_date').resample('M').sum().reset_index()
    return df

def preprocess_data_7D(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.groupby('stay_date')['room_cnt'].sum().reset_index()
    df = df.set_index('stay_date').resample('7D').sum().reset_index()
    return df

def preprocess_data_30D(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df = df.groupby('stay_date')['room_cnt'].sum().reset_index()
    df = df.set_index('stay_date').resample('30D').sum().reset_index()
    return df