import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from live_data import get_live_data
from time_series import TimeSeriesForecaster

app = Flask(__name__)
CORS(app)

FEATURES = ['PM2.5', 'PM10', 'NO2', 'CO']

# ── Load models once at startup ───────────────────────────────────────────────
try:
    with open("models.pkl", "rb") as f:
        anfis, nn, fuzzy, x_scaler, y_scaler = pickle.load(f)
    print("✅ models.pkl loaded")
    MODELS_READY = True
except FileNotFoundError:
    print("⚠️  models.pkl not found — run train.py first")
    anfis = nn = fuzzy = x_scaler = y_scaler = None
    MODELS_READY = False

# ── Time-series forecaster (lazy, cached) ─────────────────────────────────────
ts = TimeSeriesForecaster()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_models():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded — run train.py first"}), 503
    return None

def _scale_input(data):
    x = np.array([float(data.get(k, 0.0)) for k in FEATURES])
    return x_scaler.transform([x])[0]

def _denorm(val):
    return float(y_scaler.inverse_transform([[val]])[0][0])

def _level(v):
    if v > 0.7: return "High"
    if v > 0.4: return "Medium"
    return "Low"

def _validate(data):
    missing = [k for k in FEATURES if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    return None


# ── /predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    err = _require_models()
    if err: return err

    data = request.json or {}
    err  = _validate(data)
    if err: return err

    xs = _scale_input(data)

    # Fuzzy outputs real AQI directly (singletons are calibrated AQI values)
    # NN and ANFIS output normalised [0,1] — need inverse_transform
    fuzzy_aqi = float(fuzzy.predict(xs))
    nn_aqi    = round(_denorm(nn.predict(xs)), 2)
    anfis_aqi = round(_denorm(anfis.forward(xs)), 2)

    return jsonify({
        "ANFIS":       anfis_aqi,
        "NN":          nn_aqi,
        "Fuzzy":       round(fuzzy_aqi, 2),
        "explanation": {k: _level(xs[i]) for i, k in enumerate(FEATURES)},
        "inputs":      data,
    })


# ── /explain/fuzzy ────────────────────────────────────────────────────────────
@app.route("/explain/fuzzy", methods=["POST"])
def explain_fuzzy():
    err = _require_models()
    if err: return err

    xs = _scale_input(request.json or {})
    return jsonify(fuzzy.get_explanation(xs))


# ── /explain/anfis ────────────────────────────────────────────────────────────
@app.route("/explain/anfis", methods=["POST"])
def explain_anfis():
    err = _require_models()
    if err: return err

    xs = _scale_input(request.json or {})
    anfis.forward(xs)          # populates _last_activations
    return jsonify(anfis.get_rule_activations())


# ── /model-info ───────────────────────────────────────────────────────────────
@app.route("/model-info")
def model_info():
    err = _require_models()
    if err: return err

    return jsonify({
        "fuzzy": {
            "mf_params":  fuzzy.get_mf_params(),
            "n_inputs":   4,
            "design":     "Per-pollutant Mamdani sub-AQI, max aggregation",
            "singletons": fuzzy.AQI_SINGLETONS,
        },
        "nn":    nn.get_info(),
        "anfis": anfis.get_info(),
    })


# ── /live/<city> ─────────────────────────────────────────────────────────────
@app.route("/live/<city>")
def live(city):
    return jsonify(get_live_data(city))


# ── /forecast/<city> ─────────────────────────────────────────────────────────
@app.route("/forecast/<city>")
def forecast(city):
    preds = ts.predict(city, steps=24)
    return jsonify({"forecast": preds, "city": city, "steps": len(preds)})


if __name__ == "__main__":
    app.run(debug=True)