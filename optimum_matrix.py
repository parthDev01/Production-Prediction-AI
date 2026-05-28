import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import warnings
import logging
from datetime import datetime, timedelta
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')


class DynamicModelManager:
    """Tracks model performance and triggers retraining when drift is detected."""

    def __init__(self, retrain_threshold=0.1, performance_window=100):
        self.retrain_threshold = retrain_threshold
        self.performance_window = performance_window
        self.performance_history = deque(maxlen=performance_window)
        self.baseline_performance = None
        self.last_retrain = datetime.now()
        self.retrain_interval = timedelta(hours=24)

    def should_retrain(self, current_performance):
        if len(self.performance_history) < 10:
            return False
        if self.baseline_performance is None:
            self.baseline_performance = np.mean(list(self.performance_history)[:20])
        recent_performance = np.mean(list(self.performance_history)[-10:])
        performance_drop = self.baseline_performance - recent_performance
        time_since_retrain = datetime.now() - self.last_retrain
        return (performance_drop > self.retrain_threshold and
                time_since_retrain > self.retrain_interval)

    def update_performance(self, score):
        self.performance_history.append(score)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw sensor data into model-ready features.

    Handles timestamp parsing, physics-based heat transfer calculations,
    lag/rolling statistics, and resin buildup risk scoring — all using
    only past observations to avoid data leakage.
    """

    COOLING_ROLLERS = [
        'Cooling_Roller_1_Surface_Temp', 'Cooling_Roller_2_Surface_Temp',
        'Cooling_Roller_3_Surface_Temp', 'Cooling_Roller_4_Surface_Temp'
    ]

    LAG_FEATURES = [
        'Chill_Water_Inlet_Temperature', 'Chill_Water_Outlet_Temperature',
        'Chill_Water_Tank_Temp', 'Cooling_Roll_Area_Humidity',
        'Cooling_Roll_Area_Temperature', 'Cooling_Roller_Avg_Temp',
        'PID_Output_Valve_Status', 'Web Speed', 'Web Tension',
        'Web_Temp_After_Oven', 'Resin_Buildup_Risk', 'Thermal_Shock_Gradient',
        'Heat_Transfer_Rate', 'Cooling_Efficiency'
    ]

    def __init__(self, target_horizon_minutes=60):
        self.target_horizon_minutes = target_horizon_minutes
        self.median_interval = None
        self.shift_periods = None

    def fit(self, X, y=None):
        if 'Timestamp' in X.columns:
            X = self._parse_timestamps(X.copy())
            time_diffs = X['Timestamp'].diff().dt.total_seconds() / 60
            self.median_interval = float(time_diffs.median()) if time_diffs.notna().any() else 1.0
            self.shift_periods = max(1, int(self.target_horizon_minutes / max(self.median_interval, 1e-6)))
            logging.info(f"Sampling interval: {self.median_interval:.1f} min → shift={self.shift_periods} periods")
        else:
            self.median_interval = 1.0
            self.shift_periods = max(1, int(self.target_horizon_minutes))
        return self

    def transform(self, X):
        df = X.copy()
        df = self._parse_timestamps(df)

        # Time-based features
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

        # Ensure roller columns exist
        for r in self.COOLING_ROLLERS:
            if r not in df.columns:
                df[r] = np.nan

        # Physics-based features
        df['Web_Temp_After_Oven'] = df['Post_Dryer_Web_Temp_OP']
        df['Thermal_Shock_Gradient'] = (df['Post_Dryer_Web_Temp_OP'] - df['Rewinder_Web_Temp']).abs()

        heat_res = df.apply(self._heat_transfer_row, axis=1, result_type='expand')
        df['Heat_Transfer_Rate'] = heat_res[0]
        df['Resin_Buildup_Risk'] = heat_res[1]

        # Control loop features
        df['PID_Error'] = df['PID_Set_Point'] - df['Rewinder_Web_Temp']
        df['Temp_Diff_Inlet_Outlet'] = df['Chill_Water_Inlet_Temperature'] - df['Chill_Water_Outlet_Temperature']
        df['Cooling_Efficiency'] = df['Temp_Diff_Inlet_Outlet'] / (df['Chill_Water_Inlet_Flowrate'] + 1e-6)
        df['Cooling_Roller_Avg_Temp'] = df[self.COOLING_ROLLERS].mean(axis=1)
        df['Cooling_Roller_Temp_Std'] = df[self.COOLING_ROLLERS].std(axis=1)

        # Process dynamics
        df['Speed_Change_Rate'] = df['Web Speed'].diff()
        df['Tension_Stability'] = df['Web Tension'].rolling(window=5, min_periods=1).std()

        # Lag / rolling features (past-only — no leakage)
        for feature in self.LAG_FEATURES:
            if feature not in df.columns:
                df[feature] = np.nan
            for lag in [1, 2, 3]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            df[f'{feature}_roll_mean_3'] = df[feature].rolling(window=3, min_periods=1).mean()
            df[f'{feature}_roll_std_3'] = df[feature].rolling(window=3, min_periods=1).std()
            df[f'{feature}_roll_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
            df[f'{feature}_rate_change'] = df[feature].diff()

        # Fill NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        df['Speed_Change_Rate'] = df['Speed_Change_Rate'].fillna(0)
        df['Tension_Stability'] = df['Tension_Stability'].fillna(0)

        return df

    def _parse_timestamps(self, df):
        if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            ts = pd.to_datetime(df['Timestamp'], errors='coerce')
            if ts.isna().mean() > 0.5:  # likely Excel serial numbers
                ts = pd.to_datetime(df['Timestamp'], unit='D', origin='1899-12-30', errors='coerce')
            df['Timestamp'] = ts
        elif 'Timestamp' not in df.columns:
            df['Timestamp'] = pd.NaT
        return df

    def _heat_transfer_row(self, row):
        return self._calculate_heat_transfer(
            row['Web_Temp_After_Oven'],
            [row[r] for r in self.COOLING_ROLLERS],
            row['Web Speed']
        )

    def _calculate_heat_transfer(self, web_temp, roller_temps, web_speed,
                                  paper_width=1.27, paper_thickness=0.00035):
        """
        Estimates heat transfer rate (kW proxy) and resin buildup risk [0, 1].

        Uses paper mass flow rate, specific heat, and the temperature delta
        between web and cooling rollers to quantify thermal stress.
        """
        specific_heat = 1.34   # kJ/kg·K
        density = 800          # kg/m³
        gsm = density * paper_thickness
        speed_m_s = (web_speed or 0) / 60.0
        mass_flow = gsm * paper_width * speed_m_s

        avg_roller_temp = np.nanmean(roller_temps) if roller_temps else 0
        delta_T = (web_temp or 0) - (avg_roller_temp or 0)
        heat_transfer_rate = mass_flow * specific_heat * delta_T

        base_risk = min(1.0, max(0.0, (delta_T - 15) / 15))
        speed_factor = min(1.0, (web_speed or 0) / 100)
        resin_risk = min(1.0, max(0.0, base_risk * (1 + 0.3 * speed_factor)))

        return heat_transfer_rate, resin_risk


class OptimumMatrix:
    """
    Optimum Matrix — AI-driven web break prevention for paper/film manufacturing.

    Predicts the probability of a web break up to 60 minutes ahead and
    recommends parameter adjustments to keep the rewinder web temperature
    at its optimal setpoint while minimising resin buildup on cooling rollers.

    Architecture:
        - XGBoost classifier (calibrated) for break probability
        - XGBoost regressor for rewinder temperature forecasting
        - IsolationForest for anomaly filtering during training
        - SHAP for per-prediction explainability
    """

    ADJUSTABLE_PARAMS = [
        'Chill_Water_Inlet_Flowrate', 'Chill_Water_Inlet_Temperature',
        'Chill_Water_Tank_Temp', 'Zone_8_Temperature',
        'Web Tension', 'Post_Dryer_Web_Temp_OP'
    ]

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.break_model = None
        self.temp_model = None
        self.calibrated_model = None
        self.feature_names = None
        self.model_manager = DynamicModelManager()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df):
        """Train break and temperature models on labelled historical data."""
        logging.info("Starting Optimum Matrix training...")
        self.feature_engineer.fit(df)

        X_break, y_break, self.feature_names = self._prepare(df, target='Web_break', training=True)
        X_temp, y_temp, _ = self._prepare(df, target='Rewinder_Web_Temp', training=True)

        # Anomaly filtering (index-aware)
        anomaly_labels = pd.Series(
            self.anomaly_detector.fit_predict(X_break), index=X_break.index
        )
        clean_mask = anomaly_labels != -1
        logging.info(f"Removed {int((anomaly_labels == -1).sum())} anomalous training samples")

        X_break_clean = X_break.loc[clean_mask]
        y_break_clean = y_break.loc[clean_mask]

        # Align temp data to same index
        X_temp_aligned = X_temp.reindex(X_break.index)
        y_temp_aligned = y_temp.reindex(X_break.index)
        valid_temp = X_temp_aligned.notna().any(axis=1)
        combined_mask = clean_mask & valid_temp

        X_temp_clean = X_temp_aligned.loc[combined_mask]
        y_temp_clean = y_temp_aligned.loc[combined_mask]
        X_break_clean = X_break_clean.reindex(X_temp_clean.index)
        y_break_clean = y_break_clean.reindex(X_temp_clean.index)

        self.scaler.fit(X_break_clean)
        X_b_scaled = self._scale(X_break_clean)
        X_t_scaled = self._scale(X_temp_clean)

        self.break_model = self._train_break_model(X_b_scaled, y_break_clean.astype(int))
        self.temp_model = self._train_temp_model(X_t_scaled, y_temp_clean)

        self.calibrated_model = CalibratedClassifierCV(self.break_model, method='isotonic', cv=3)
        self.calibrated_model.fit(X_b_scaled, y_break_clean.astype(int))

        logging.info("Training complete.")

    def _train_break_model(self, X, y):
        pos = max(1, int((y == 1).sum()))
        neg = max(1, int((y == 0).sum()))

        model = xgb.XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=max(1.0, neg / pos),
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        param_grid = {
            'n_estimators': [150, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5),
                            scoring='f1', n_jobs=-1, verbose=1)
        grid.fit(X, y)
        logging.info(f"Break model — best params: {grid.best_params_}, F1: {grid.best_score_:.3f}")
        return grid.best_estimator_

    def _train_temp_model(self, X, y):
        model = xgb.XGBRegressor(
            random_state=42, n_jobs=-1, tree_method='hist', eval_metric='rmse'
        )
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.1]
        }
        grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5),
                            scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X, y)
        logging.info(f"Temp model — best params: {grid.best_params_}, MSE: {-grid.best_score_:.3f}")
        return grid.best_estimator_

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df):
        """
        Return break probabilities, temperature forecasts, and SHAP explanations.

        Returns a dict with keys:
            break_predictions  : np.ndarray[int]   — binary (0/1)
            break_probabilities: np.ndarray[float] — calibrated probability [0, 1]
            temp_predictions   : np.ndarray[float] — rewinder temp forecast (°C)
            top_factors        : list[str]          — highest-SHAP feature per row
            shap_values        : np.ndarray         — full SHAP matrix
        """
        X, _, _ = self._prepare(df, training=False)
        X_scaled = self._scale(X)

        break_pred = self.calibrated_model.predict(X_scaled)
        break_prob = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        temp_pred = self.temp_model.predict(X_scaled)

        shap_array = self._compute_shap(X_scaled)
        top_factors = [
            self.feature_names[int(np.argmax(np.abs(shap_array[i])))]
            for i in range(shap_array.shape[0])
        ]

        return {
            'break_predictions': break_pred,
            'break_probabilities': break_prob,
            'temp_predictions': temp_pred,
            'top_factors': top_factors,
            'shap_values': shap_array
        }

    def recommend(self, df, target_temp=33.5, max_break_prob=0.1):
        """
        For each row, sweep adjustable parameters to find the combination that
        minimises a composite score: temperature deviation + break risk + change magnitude.
        """
        X, _, _ = self._prepare(df, training=False)
        return [self._optimise_row(X.iloc[i:i+1].copy(), target_temp, max_break_prob)
                for i in range(len(X))]

    def _optimise_row(self, row, target_temp, max_break_prob):
        best, best_score = {}, float('inf')
        for param in self.ADJUSTABLE_PARAMS:
            if param not in row.columns:
                continue
            current = float(row[param].iloc[0])
            lo = current * 0.9 if current != 0 else -0.1
            hi = current * 1.1 if current != 0 else 0.1
            for val in np.linspace(lo, hi, 5):
                test = row.copy()
                test[param] = val
                t_scaled = self._scale(test)
                bp = float(self.calibrated_model.predict_proba(t_scaled)[0, 1])
                tp = float(self.temp_model.predict(t_scaled)[0])
                score = (abs(tp - target_temp) * 10 +
                         max(0.0, bp - max_break_prob) * 100 +
                         abs(val - current) * 0.1)
                if score < best_score:
                    best_score = score
                    best[param] = {'current': current, 'recommended': float(val),
                                   'change': float(val - current)}
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, df, target='Web_break', training=True):
        df_feat = self.feature_engineer.transform(df)
        exclude = ['Timestamp']
        if training:
            exclude += ['Web_break', 'Rewinder_Web_Temp']

        if training:
            features = [c for c in df_feat.columns if c not in exclude]
            self.feature_names = features
            X = df_feat[features].copy()
            shift = self.feature_engineer.shift_periods or 2
            if target == 'Web_break':
                y = df['Web_break'].shift(-shift).ffill().astype(float)
            else:
                y = df['Rewinder_Web_Temp'].shift(-shift)
            valid = y.notnull()
            return X.loc[valid], y[valid].astype(int if target == 'Web_break' else float), features
        else:
            if self.feature_names is None:
                self.feature_names = [c for c in df_feat.columns if c not in exclude]
            for c in self.feature_names:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            return df_feat[self.feature_names].copy(), None, self.feature_names

    def _scale(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    def _compute_shap(self, X_scaled):
        try:
            explainer = shap.TreeExplainer(self.break_model)
            shap_vals = explainer.shap_values(X_scaled)
        except Exception:
            sv = shap.Explainer(self.break_model)(X_scaled)
            shap_vals = getattr(sv, 'values', sv)
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        return arr
