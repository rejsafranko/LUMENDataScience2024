import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
import plotly.graph_objs as go
from plotly.offline import plot

# Ignore specific warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names, but DecisionTreeRegressor was fitted without feature names")

def get_data_for_prediction(df):
    features = [col for col in df.columns if col not in ['stay_date', 'room_cnt']]
    X = df[features]
    y = df['room_cnt']
    return df, X, y

def predict_intervals(model, X_test, percentile=95):
    predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
    lower_bound = np.percentile(predictions, (100 - percentile) / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2, axis=0)
    return lower_bound, upper_bound

def plot_results(X_test_dates, y_test, y_pred, lower_bound, upper_bound):
    trace1 = go.Scatter(x=X_test_dates, y=y_test, mode='lines', name='Actual')
    trace2 = go.Scatter(x=X_test_dates, y=y_pred, mode='lines', name='Predicted')
    trace3 = go.Scatter(x=X_test_dates, y=lower_bound, mode='lines', name='Lower Bound', fill=None)
    trace4 = go.Scatter(x=X_test_dates, y=upper_bound, mode='lines', name='Upper Bound', fill='tonexty')

    layout = go.Layout(
        title='Room Count Forecast with Prediction Intervals',
        xaxis_title='Date',
        yaxis_title='Room Count',
        showlegend=True
    )
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def evaluate_precision(y_test, lower_bound, upper_bound):
    within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean()
    result = f"Percentage of actual values within predicted interval: {within_interval * 100:.2f}%"
    return result

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = (mse ** 0.5)
    results = {
        'Mean Squared Error': f"{mse:.2f}",
        'Root Mean Squared Error': f"{rmse:.2f}"
    }
    return results

