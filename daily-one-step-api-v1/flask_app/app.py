from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from joblib import dump, load
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

#features of saved model - just save features during training and load them here
required_columns_for_regression_next_day_prediction = ['stay_day_of_week', 'total_rooms_lag1', 'stay_is_weekend', 'stay_week_of_year', 'rooms_reserved_that_day_lag1']
required_columns_for_cancellation_prediction = ['night_number','room_cnt', 'total_price', 'room_category_id', 'sales_channel_id', 'stay_day_of_week', 'stay_month', 'stay_day_of_month', 'stay_is_weekend', 'stay_quarter', 'stay_week_of_year', 'guest_country_id_encoded', 'lead_time', 'total_guests']


def prepare(df):
    date_columns = ['stay_date', 'reservation_date', 'date_from', 'date_to', 'cancel_date']
    df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')

    df.drop(columns=df.filter(regex='tax').columns, inplace=True)
    
    # Impute missing values for specified columns using KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    columns_to_impute = ['price', 'total_price', 'food_price', 'other_price']
    df[columns_to_impute] = knn_imputer.fit_transform(df[columns_to_impute])

    filter1 = (df.children_cnt == 0) & (df.adult_cnt == 0)
    filter2 = df['reservation_date'] > df['date_from']
    filter3 = df['date_from'] < df['cancel_date']
    df = df[~(filter1 | filter2 | filter3)]

    df['stay_day_of_week'] = df['stay_date'].dt.dayofweek
    df['stay_month'] = df['stay_date'].dt.month
    df['stay_year'] = df['stay_date'].dt.year
    df['stay_day_of_month'] = df['stay_date'].dt.day
    df['stay_is_weekend'] = (df['stay_day_of_week'] > 4).astype(int)
    df['stay_quarter'] = df['stay_date'].dt.quarter
    df['stay_week_of_year'] = df['stay_date'].dt.isocalendar().week
    df['cancelled_days_after_reservation'] = (df['cancel_date'] - df['reservation_date']).dt.days
    df['cancelled_days_before_arrival'] = (df['date_from'] - df['cancel_date']).dt.days

    le = LabelEncoder()
    df['guest_country_id_encoded'] = le.fit_transform(df['guest_country_id'])
    df['reservation_status_encoded'] = le.fit_transform(df['reservation_status'])

    df.drop(columns=['guest_country_id', 'reservation_status', 'resort_id'], inplace=True)
    df['lead_time'] = (df['date_from'] - df['reservation_date']).dt.days
    df['total_guests'] = df['children_cnt'] + df['adult_cnt']
    useless_col = ['adult_cnt', 'stay_year', 'children_cnt', 'food_price', 'other_price', 'price', 'reservation_id', 'guest_id']
    df.drop(useless_col, axis=1, inplace=True)

    df['reservation_status_encoded'] = df['reservation_status_encoded'].apply(lambda x: 0 if x in [0, 2] else x)
    filtered_df = df[df['reservation_status_encoded'] == 1]

    return df, filtered_df


def prepare_regression(filtered_df):
    aggregations = {
    'room_cnt': [
        ('total_rooms', 'sum'),
        ('rooms_reserved_that_day', lambda x: (filtered_df.loc[x.index, 'reservation_date'] == filtered_df.loc[x.index, 'stay_date']).sum())
    ],
    'total_price': [('average_room_price','mean')],
    'lead_time': [('average_lead_time','mean')],
    'room_category_id': [('room_category', lambda x: x.mode()[0] if not x.mode().empty else None)],
    'sales_channel_id': [('sales_channel_mode', lambda x: x.mode()[0] if not x.mode().empty else None)],
    'stay_day_of_week': [('stay_day_of_week', 'min')],
    'stay_month': [('stay_of_month', 'min')],
    'stay_day_of_month': [('stay_day_of_month', 'min')],
    'stay_is_weekend': [('stay_is_weekend', 'min')],
    'stay_quarter': [('stay_quarter', 'min')],
    'stay_week_of_year': [('stay_week_of_year', 'min')],
    }
    grouped = filtered_df.groupby('stay_date').agg(aggregations)
    grouped.columns = grouped.columns.droplevel(0)
    grouped = grouped.reset_index()
    grouped.rename(columns={'index': 'stay_date'}, inplace=True)
    grouped = grouped.sort_index()
    for col in ['rooms_reserved_that_day', 'total_rooms', 'average_room_price', 'average_lead_time', 'room_category', 'sales_channel_mode']:
        grouped[f'{col}_lag1'] = grouped[col].shift(1)
    regression_predict = grouped
    return regression_predict


def predict(df, regression_predict):
    clf_model = load('random_forest_classifier.joblib')
    reg_model = load('random_forest_regressor.joblib')
    
    dates_to_process = df['stay_date'].unique()
    
    # Sort dates to ensure the ordering for shift operations later
    dates_to_process = np.sort(dates_to_process)
    
    sigma = 2.93
    results = []
    
    for stay_date in dates_to_process:
        data_for_prediction = df[(df['reservation_date'] < stay_date) & (df['reservation_status_encoded'] == 1) & (df['stay_date'] == stay_date)]
        data_for_regression = regression_predict[regression_predict['stay_date'] == stay_date]
        
        if data_for_prediction.empty or data_for_regression.empty:
            continue
        
        # Select required columns earlier to avoid repetitive DataFrame copying
        data_for_prediction = data_for_prediction[required_columns_for_cancellation_prediction] 
        data_for_regression = data_for_regression[required_columns_for_regression_next_day_prediction]
        
        # Perform predictions
        predicted_status = clf_model.predict(data_for_prediction)
        data_for_prediction['predicted_status'] = predicted_status
        total_rooms = data_for_prediction[data_for_prediction['predicted_status'] == 1]['room_cnt'].sum()

        # Regression predictions and interval calculations
        predicted_rooms = reg_model.predict(data_for_regression)[0]
        lower_bound = predicted_rooms - 1.96 * sigma
        upper_bound = predicted_rooms + 1.96 * sigma
        
        results.append({
            'stay_date': stay_date,
            'exact_rooms_predicted': int(total_rooms + predicted_rooms),
            'lower_bound_rooms_predicted': max(0, int(total_rooms + lower_bound)),
            'upper_bound_rooms_predicted': int(total_rooms + upper_bound)
        })
    
    # Creating DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df


@app.route('/', methods=['GET'])
def index():
    # Render an HTML form to upload the .parquet file
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.parquet'):
        # Load the .parquet file into a DataFrame
        df = pd.read_parquet(file)
        df, filtered_df = prepare(df)
        regression_predict = prepare_regression(filtered_df)
        results_df = predict(df, regression_predict)

        results_df = results_df.merge(regression_predict[['stay_date', 'total_rooms']], on='stay_date', how='left')
        results_df = results_df.rename(columns={
            'stay_date': 'stay_date',
            'lower_bound_rooms_predicted': 'lower_bound',
            'upper_bound_rooms_predicted': 'upper_bound',
            'exact_rooms_predicted': 'predicted_rooms',
            'total_rooms': 'total_rooms'
        })
        results_df = results_df[['stay_date', 'lower_bound', 'upper_bound', 'predicted_rooms', 'total_rooms']]
        results_df.to_csv("results.csv", index=False)

        plt.figure(figsize=(10, 6))

        # Plotting the predicted values as a line
        plt.plot(results_df['stay_date'], results_df['predicted_rooms'], color='blue', label='Predicted Total Rooms')

        # Plotting real values as a line
        plt.plot(results_df['stay_date'], results_df['total_rooms'], color='red', label='Real Total Rooms')

        # Adding filled area for prediction bounds
        plt.fill_between(results_df['stay_date'], results_df['lower_bound'], results_df['upper_bound'], color='gray', alpha=0.3, label='Prediction Bounds')

        plt.xlabel('Stay Date')
        plt.ylabel('Total Rooms')
        plt.title('Comparison of Predicted and Real Total Rooms')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate dates for better readability if necessary
        plt.tight_layout()  # Adjust layout to make room for rotated date labels
        plt.show()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        return send_file(img, mimetype='image/png')

    return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)