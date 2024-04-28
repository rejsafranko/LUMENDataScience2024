from .utilities.feature_engineering import feature_engineering_month, calculate_aggregates, calculate_expected_room_count30D, merge_dataframes, resample_data30D
from .utilities.prediction import get_data_for_prediction, predict_intervals, plot_results, evaluate_precision, evaluate_model
import joblib
import os

def run_predict_month_one_step(df):
    df['lead_time_days'] = (df['date_from'] - df['reservation_date']).dt.days
    
    script_dir = os.path.dirname(__file__)
    rel_path = "models//month_one_step.joblib"
    abs_file_path = os.path.join(script_dir, rel_path)
    model = joblib.load(abs_file_path)

    grouped_df = calculate_aggregates(df)
    expected_room_cnt = calculate_expected_room_count30D(df)
    final_result = merge_dataframes(grouped_df, expected_room_cnt)
    processed_data = feature_engineering_month(resample_data30D(final_result))
    _, X_test, y_test = get_data_for_prediction(processed_data)
    X_test.drop('year', axis=1, inplace=True)
    y_pred = model.predict(X_test)
    lower_bound, upper_bound = predict_intervals(model, X_test)

    plot_html = plot_results(processed_data['stay_date'], y_test, y_pred, lower_bound, upper_bound)
    precision_result = evaluate_precision(y_test, lower_bound, upper_bound)
    evaluation_results = evaluate_model(y_test, y_pred)
    return plot_html, precision_result, evaluation_results
