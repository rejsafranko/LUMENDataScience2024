from .utilities.preprocessing import preprocess_data_week
from .utilities.feature_engineering import feature_engineering_week
from .utilities.prediction import get_data_for_prediction, predict_intervals, plot_results, evaluate_precision, evaluate_model
import joblib
import os

def run_predict_week_multi_step(df):
    script_dir = os.path.dirname(__file__)
    rel_path = "models//week_multi_step.joblib"
    abs_file_path = os.path.join(script_dir, rel_path)
    model = joblib.load(abs_file_path)
    df = preprocess_data_week(df)
    df = feature_engineering_week(df)
    _, X_test, y_test = get_data_for_prediction(df)
    y_pred = model.predict(X_test)
    lower_bound, upper_bound = predict_intervals(model, X_test)
    plot_html = plot_results(df['stay_date'], y_test, y_pred, lower_bound, upper_bound)
    precision_result = evaluate_precision(y_test, lower_bound, upper_bound)
    evaluation_results = evaluate_model(y_test, y_pred)
    return plot_html, precision_result, evaluation_results
