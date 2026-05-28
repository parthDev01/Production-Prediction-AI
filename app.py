"""
Production Predict — Flask Dashboard

Run with:  python app.py
           Then open http://localhost:5000
"""

import os
import logging
import argparse
import pandas as pd
from flask import Flask, render_template, request, jsonify
from production_predict import OptimumMatrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
predictor = OptimumMatrix()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    'Timestamp', 'Chill_Water_Inlet_Flowrate', 'Chill_Water_Inlet_Temperature',
    'Chill_Water_Outlet_Temperature', 'Chill_Water_Tank_Temp',
    'Cooling_Roll_Area_Humidity', 'Cooling_Roll_Area_Temperature',
    'Cooling_Roller_1_Surface_Temp', 'Cooling_Roller_2_Surface_Temp',
    'Cooling_Roller_3_Surface_Temp', 'Cooling_Roller_4_Surface_Temp',
    'PID_Output_Valve_Status', 'PID_Set_Point', 'Rewinder_Web_Temp',
    'Zone_8_Temperature', 'Web Speed', 'Web Tension', 'Post_Dryer_Web_Temp_OP'
]


def load_data(file_path, is_training=False):
    df = pd.read_excel(file_path)

    expected = REQUIRED_COLUMNS + (['Web_break'] if is_training else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.drop_duplicates(subset=['Timestamp'], keep='first')

    for col in df.columns:
        if col != 'Timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    logging.info(f"Loaded {file_path}: {df.shape[0]} rows")
    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def dashboard():
    train_path = app.config.get('TRAIN_DATA', 'data/train.xlsx')
    test_path = app.config.get('TEST_DATA', 'data/test.xlsx')
    try:
        df_train = load_data(train_path, is_training=True)
        predictor.train(df_train)

        df_test = load_data(test_path, is_training=False)
        preds = predictor.predict(df_test)
        recommendations = predictor.recommend(df_test)

        results = []
        for i in range(len(df_test)):
            results.append({
                'timestamp': df_test['Timestamp'].iloc[i],
                'break_prediction': bool(preds['break_predictions'][i]),
                'break_probability': float(preds['break_probabilities'][i]),
                'temp_prediction': float(preds['temp_predictions'][i]),
                'top_factor': preds['top_factors'][i],
                'recommendations': recommendations[i]
            })

        return render_template('dashboard.html', results=results)

    except Exception as e:
        logging.exception("Dashboard error")
        return f"<pre>Error: {e}</pre>", 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept JSON sensor readings and return break probability + recommendations."""
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
        preds = predictor.predict(df)
        recs = predictor.recommend(df)
        return jsonify({
            'break_probability': float(preds['break_probabilities'][0]),
            'break_predicted': bool(preds['break_predictions'][0]),
            'temp_forecast': float(preds['temp_predictions'][0]),
            'top_factor': preds['top_factors'][0],
            'recommendations': recs[0]
        })
    except Exception as e:
        logging.exception("API predict error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """Trigger model retrain if performance drift is detected."""
    try:
        data = request.get_json(force=True)
        performance_score = data.get('performance_score', 0.85)
        if predictor.model_manager.should_retrain(performance_score):
            df_new = pd.DataFrame(data.get('records', []))
            predictor.train(df_new)
            predictor.model_manager.last_retrain = __import__('datetime').datetime.now()
            return jsonify({'status': 'retrained'})
        return jsonify({'status': 'current'})
    except Exception as e:
        logging.exception("Retrain error")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Production Predict Dashboard')
    parser.add_argument('--train', default='data/train.xlsx', help='Path to training data (.xlsx)')
    parser.add_argument('--test', default='data/test.xlsx', help='Path to test/live data (.xlsx)')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.config['TRAIN_DATA'] = args.train
    app.config['TEST_DATA'] = args.test

    app.run(debug=True, host=args.host, port=args.port)
