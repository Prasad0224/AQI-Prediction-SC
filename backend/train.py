import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import clean_city_day, FEATURES, TARGET
from model import ANFIS, NNModel, FuzzyModel

print("=" * 55)
print("  AQI Prediction — Soft Computing Training Pipeline")
print("=" * 55)

# ── Load & preprocess data ─────────────────────────────────────────────────
print("\n[1/5] Loading data...")
df = clean_city_day("../data/city_day.csv")
print(f"      Samples: {len(df)}")

X = df[FEATURES].values
y = df[TARGET].values

# ── Scale features ────────────────────────────────────────────────────────
x_scaler = MinMaxScaler()
X_scaled  = x_scaler.fit_transform(X)

# ── Scale target (save scaler for de-normalisation in API) ────────────────
y_scaler  = MinMaxScaler()
y_scaled  = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

# ── Train / test split ────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(f"      Train: {len(X_tr)}  |  Test: {len(X_te)}")

# ── 1) Neural Network ─────────────────────────────────────────────────────
print("\n[2/5] Training Neural Network (MLP 4→64→32→1)...")
nn = NNModel()
nn_metrics = nn.train(X_tr, y_tr, X_te, y_te)
print(f"      Train RMSE : {nn_metrics['train_rmse']:.4f}")
print(f"      Test  RMSE : {nn_metrics['val_rmse']:.4f}")
print(f"      R²         : {nn_metrics['r2']:.4f}")

# ── 2) ANFIS ──────────────────────────────────────────────────────────────
print("\n[3/5] Training ANFIS (hybrid learning: LSE + GD)...")
anfis = ANFIS(n_inputs=4, n_rules=6)
anfis.train(X_tr, y_tr, epochs=60, lr=0.005)

y_hat_anfis = np.array([anfis.forward(x) for x in X_te])
anfis_rmse  = float(np.sqrt(mean_squared_error(y_te, y_hat_anfis)))
anfis_r2    = float(r2_score(y_te, y_hat_anfis))
print(f"      Test  RMSE : {anfis_rmse:.4f}")
print(f"      R²         : {anfis_r2:.4f}")

# ── 3) Fuzzy Logic ────────────────────────────────────────────────────────
print("\n[4/5] Evaluating Fuzzy Logic (Mamdani, 9 rules)...")
fuzzy = FuzzyModel()

# Fuzzy outputs real AQI — compare against de-normalised y_te for fair metrics
y_te_real        = y_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
y_hat_fuzzy      = np.array([fuzzy.predict(x) for x in X_te])
fuzzy_rmse       = float(np.sqrt(mean_squared_error(y_te_real, y_hat_fuzzy)))
fuzzy_r2         = float(r2_score(y_te_real, y_hat_fuzzy))
print(f"      Test  RMSE : {fuzzy_rmse:.2f} (real AQI units)")
print(f"      R²         : {fuzzy_r2:.4f}")

# De-normalise NN and ANFIS predictions for unified comparison table
y_hat_nn_real    = y_scaler.inverse_transform(
    np.array([nn.predict(x) for x in X_te]).reshape(-1, 1)
).ravel()
y_hat_anfis_real = y_scaler.inverse_transform(y_hat_anfis.reshape(-1, 1)).ravel()
nn_rmse_real     = float(np.sqrt(mean_squared_error(y_te_real, y_hat_nn_real)))
anfis_rmse_real  = float(np.sqrt(mean_squared_error(y_te_real, y_hat_anfis_real)))
nn_r2_real       = float(r2_score(y_te_real, y_hat_nn_real))
anfis_r2_real    = float(r2_score(y_te_real, y_hat_anfis_real))

# ── Summary table ─────────────────────────────────────────────────────────
print("\n[5/5] Results Summary  (all RMSE in real AQI units)")
print("  " + "-" * 44)
print(f"  {'Model':<12} {'Test RMSE':>12}  {'R²':>8}")
print("  " + "-" * 44)
print(f"  {'Neural Net':<12} {nn_rmse_real:>12.2f}  {nn_r2_real:>8.4f}")
print(f"  {'ANFIS':<12} {anfis_rmse_real:>12.2f}  {anfis_r2_real:>8.4f}")
print(f"  {'Fuzzy Logic':<12} {fuzzy_rmse:>12.2f}  {fuzzy_r2:>8.4f}")
print("  " + "-" * 44)

# ── Save ──────────────────────────────────────────────────────────────────
with open("models.pkl", "wb") as f:
    pickle.dump((anfis, nn, fuzzy, x_scaler, y_scaler), f)

print("\n✅ models.pkl saved successfully.\n")