from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
from predict.utilities.preprocessing import preprocess_data_day, clean
from predict.predict_day_multi_step import run_predict_day_multi_step
from predict.predict_week_multi_step import run_predict_week_multi_step
from predict.predict_month_multi_step import run_predict_month_multi_step
from predict.predict_week_one_step import run_predict_week_one_step
from predict.predict_month_one_step import run_predict_month_one_step



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if file and (filename.endswith('.csv') or filename.endswith('.parquet')):
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith('.parquet'):
                df = pd.read_parquet(file)

            processed_df = clean(df)
            results = []

            # Daily predictions
            plot_html_day, precision_day, eval_results_day = run_predict_day_multi_step(processed_df)
            results.append({
                'title': 'Daily Predictions MultiStep',
                'plot_html': plot_html_day,
                'precision_result': precision_day,
                'evaluation_results': eval_results_day
            })

            # Weekly predictions
            plot_html_week, precision_week, eval_results_week = run_predict_week_multi_step(processed_df)
            results.append({
                'title': 'Weekly Predictions MultiStep',
                'plot_html': plot_html_week,
                'precision_result': precision_week,
                'evaluation_results': eval_results_week
            })

            # Weekly predictions One step
            plot_html_week_onestep, precision_week_onestep, eval_results_week_onestep = run_predict_week_one_step(processed_df)
            results.append({
                'title': 'Weekly Predictions One Step',
                'plot_html': plot_html_week_onestep,
                'precision_result': precision_week_onestep,
                'evaluation_results': eval_results_week_onestep
            })

            # Monthly predictions
            plot_html_month, precision_month, eval_results_month = run_predict_month_multi_step(processed_df)
            results.append({
                'title': 'Monthly Predictions MultiStep',
                'plot_html': plot_html_month,
                'precision_result': precision_month,
                'evaluation_results': eval_results_month
            })

            # Weekly predictions One step
            plot_html_month_onestep, precision_month_onestep, eval_results_month_onestep = run_predict_month_one_step(processed_df)
            results.append({
                'title': 'Monthly Predictions One Step',
                'plot_html': plot_html_month_onestep,
                'precision_result': precision_month_onestep,
                'evaluation_results': eval_results_month_onestep
            })

            return render_template('results.html', results=results)
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    app.run(debug=True, port=1234)
