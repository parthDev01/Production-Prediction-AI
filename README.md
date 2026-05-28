# Production Predict

**AI-driven web break prevention for paper and film manufacturing.**

Production Predict monitors 15+ real-time sensor streams across the dryer and cooling section of a converting line, predicts the probability of a web break up to 60 minutes ahead, and tells operators the exact parameter adjustments needed to prevent it — before resin has a chance to build up on the cooling rollers.

---

## Results

Deployed and validated in a live production environment over a 6-month period (January–August).

| Metric | Before | After | Change |
|---|---|---|---|
| Resin/rewinder downtime | 2,780 min | 1,150 min | **−58.6%** |
| Share of total downtime | 35.1% | 14.2% | **−20.9%** |

> Total line downtime remained consistent across both periods, isolating the improvement specifically to the resin/rewinder failure mode — the targeted problem.

---

## The Problem

In paper and film converting, the web (the moving sheet) exits the dryer at elevated temperatures and passes over a series of chilled cooling rollers before reaching the rewinder. If the rewinder web temperature rises above its setpoint, resin from the coating migrates onto the roller surfaces. Over time this buildup causes the web to stick, creating a web break — an unplanned stoppage that requires a full line shutdown, cleanup, and restart.

Operators have multiple levers to control web temperature (chilled water flowrate, valve position, web speed, dryer zone temperatures), but the thermal dynamics are non-linear and the effects are delayed. By the time an operator notices the temperature climbing, resin has often already started to accumulate.

---

## How It Works

### 1. Physics-Based Feature Engineering

Raw sensor readings are transformed into features that encode the underlying thermal physics of the process:

- **Heat Transfer Rate** — estimated kW transferred from the web to the cooling rollers, derived from web mass flow rate, specific heat of paper, and the temperature delta across the cooling section
- **Resin Buildup Risk** — a [0, 1] score combining temperature delta and web speed; high speed at high delta-T accelerates deposition
- **Thermal Shock Gradient** — temperature difference between the post-dryer web and the rewinder, a proxy for the instantaneous cooling demand
- **Cooling Efficiency** — chilled water temperature rise per unit of flowrate

On top of these, the model computes lag features (1, 2, 3 readings back), 3- and 5-period rolling means and standard deviations, and rate-of-change for all key process variables — giving the model a picture of *trends*, not just instantaneous values.

### 2. Predictive Models

Two XGBoost models are trained end-to-end on labelled historical data:

| Model | Target | Horizon | Validation |
|---|---|---|---|
| Break classifier | Web break probability | 60 min ahead | Time-series CV (5 folds), calibrated with isotonic regression |
| Temperature regressor | Rewinder web temp | 60 min ahead | Time-series CV (5 folds) |

Training uses `TimeSeriesSplit` throughout to prevent any data leakage. An `IsolationForest` pre-filters anomalous training samples (sensor faults, startup transients) before fitting.

### 3. Operator Recommendations

For each prediction, the system sweeps six adjustable parameters (±10% of current values) and finds the combination that minimises a composite score:

```
score = 10 × |temp_forecast − setpoint| + 100 × max(0, break_prob − threshold) + 0.1 × |parameter_change|
```

The result is a ranked table of recommended adjustments — concrete, actionable numbers the operator can dial in immediately.

### 4. Explainability

Every prediction is accompanied by a SHAP breakdown identifying which sensor or derived feature drove the risk score. This builds operator trust and makes it easy to spot when a sensor is faulty rather than the process genuinely deteriorating.

---

## Architecture

```
data/
├── train.xlsx          # Historical data with Web_break labels
└── test.xlsx           # Live / holdout data

optimum_matrix.py       # Core model: FeatureEngineer + OptimumMatrix class
app.py                  # Flask dashboard and REST API
templates/
└── dashboard.html      # Real-time operator dashboard
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live dashboard (trains on startup) |
| `POST` | `/api/predict` | Single-reading inference (JSON in, JSON out) |
| `POST` | `/api/retrain` | Trigger retrain if performance drift detected |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/your-username/optimum-matrix.git
cd optimum-matrix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
#    See data/sample_schema.md for the expected column list
cp your_training_data.xlsx data/train.xlsx
cp your_live_data.xlsx     data/test.xlsx

# 4. Run
python app.py --train data/train.xlsx --test data/test.xlsx --port 5000
```

Open `http://localhost:5000` to view the dashboard.

### Predict via API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Timestamp": "2024-06-15 09:30:00",
    "Chill_Water_Inlet_Flowrate": 45.2,
    "Chill_Water_Inlet_Temperature": 12.1,
    "Chill_Water_Outlet_Temperature": 16.8,
    "Rewinder_Web_Temp": 34.7,
    "Web Speed": 82.0,
    "Web Tension": 210.5,
    "Post_Dryer_Web_Temp_OP": 95.3,
    "PID_Set_Point": 33.5
  }'
```

---

## Sensor Inputs (15+ data points)

| Category | Sensors |
|---|---|
| Cooling water | Inlet temp, outlet temp, tank temp, flowrate |
| Cooling rollers | Surface temp (×4), area temperature, area humidity |
| Web properties | Speed, tension, post-dryer temperature, rewinder temperature |
| Control system | PID set point, PID valve output |
| Dryer | Zone 8 temperature |

See [`data/sample_schema.md`](data/sample_schema.md) for the full column specification.

---

## Tech Stack

- **ML** — XGBoost, scikit-learn (IsolationForest, CalibratedClassifierCV, TimeSeriesSplit)
- **Explainability** — SHAP (TreeExplainer)
- **Backend** — Flask
- **Data** — pandas, NumPy, openpyxl
