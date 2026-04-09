import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# FUZZY LOGIC MODEL  (Improved — all 4 pollutants, calibrated MFs)
# Mamdani Fuzzy Inference System
#   - Gaussian MFs calibrated to actual training-data percentiles
#   - Per-pollutant independent sub-AQI inference
#   - Final AQI = max of all sub-AQIs (mirrors how real AQI is computed)
#   - Inputs: PM2.5 (x[0]), PM10 (x[1]), NO2 (x[2]), CO (x[3])
# =============================================================================

class FuzzyModel:
    """
    Improved Mamdani FIS that uses ALL 4 pollutants.

    Why this design:
      · Each pollutant independently infers a sub-AQI through its own
        Low/Medium/High membership functions and centroid defuzzification.
      · Final AQI = max(sub-AQIs) — exactly how the real AQI standard works:
        the dominant (worst) pollutant determines the overall index.
      · MF centers are calibrated from actual training-data percentiles so
        the Low/Medium/High boundaries reflect real pollution distributions.
    """

    # Flag: output is real AQI, NOT normalised [0,1]
    OUTPUT_IS_NORMALIZED = False

    FEATURE_NAMES = ['PM2.5', 'PM10', 'NO2', 'CO']

    # AQI singleton values for each linguistic level
    # Calibrated to centre of each Indian AQI band
    AQI_SINGLETONS = {
        'Good':         25,    # (0-50)
        'Satisfactory': 75,    # (51-100)
        'Moderate':     150,   # (101-200)
        'Poor':         250,   # (201-300)
        'Severe':       350,   # (301-400+)
    }

    def __init__(self):
        # Default centres & sigmas — evenly spaced; overwritten by calibrate()
        self._centers = np.tile([0.1, 0.3, 0.5, 0.7, 0.9], (4, 1))   # (4, 5)
        self._sigmas  = np.ones((4, 5)) * 0.1

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, X_scaled):
        """
        Set MF centres and widths from actual training-data distribution.
        Uses percentiles so boundaries cover the real spread of each pollutant.
        """
        p_vals = [np.percentile(X_scaled, p, axis=0) for p in [20, 40, 60, 80]]

        for i in range(4):
            bounds = [float(p[i]) for p in p_vals]
            min_val, max_val = float(X_scaled[:, i].min()), float(X_scaled[:, i].max())

            self._centers[i] = [
                (min_val + bounds[0]) / 2,
                (bounds[0] + bounds[1]) / 2,
                (bounds[1] + bounds[2]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[3] + max_val) / 2,
            ]
            
            # Distance between centers as sigma width
            for j in range(5):
                left_c = self._centers[i][max(0, j-1)]
                right_c = self._centers[i][min(4, j+1)]
                self._sigmas[i][j] = max(abs(right_c - left_c) / 3.0, 0.1)

    # ── MF helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian(x, c, sigma):
        return float(np.exp(-((x - c) ** 2) / (2 * sigma ** 2)))

    def _memberships(self, xi, i):
        """Membership for feature i at normalized value xi."""
        labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Severe']
        return {
            lbl: self._gaussian(xi, self._centers[i][j], self._sigmas[i][j])
            for j, lbl in enumerate(labels)
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def _sub_aqi(self, xi, i):
        """Centroid defuzzification → sub-AQI for pollutant i."""
        mu  = self._memberships(xi, i)
        num = sum(mu[lbl] * self.AQI_SINGLETONS[lbl] for lbl in mu)
        den = sum(mu.values())
        return num / den if den > 1e-9 else self.AQI_SINGLETONS['Good']

    def predict(self, x):
        """Final AQI = max of 4 per-pollutant sub-AQIs."""
        sub_aqis = [self._sub_aqi(float(x[i]), i) for i in range(4)]
        return float(max(sub_aqis))

    # ── Explainability ────────────────────────────────────────────────────────

    def get_explanation(self, x):
        memberships = {}
        sub_aqis    = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            mu = self._memberships(float(x[i]), i)
            memberships[name] = {k: round(v, 4) for k, v in mu.items()}
            sub_aqis[name]    = round(self._sub_aqi(float(x[i]), i), 2)

        dominant = max(sub_aqis, key=sub_aqis.get)
        return {
            'memberships':       memberships,
            'sub_aqis':          sub_aqis,
            'dominant_pollutant': dominant,
            'mf_centers': {
                self.FEATURE_NAMES[i]: {
                    'Good':   round(float(self._centers[i][0]), 4),
                    'Sat':    round(float(self._centers[i][1]), 4),
                    'Mod':    round(float(self._centers[i][2]), 4),
                    'Poor':   round(float(self._centers[i][3]), 4),
                    'Severe': round(float(self._centers[i][4]), 4),
                }
                for i in range(4)
            },
        }

    def get_mf_params(self):
        labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Severe']
        return {
            self.FEATURE_NAMES[i]: {
                lbl: {
                    'c':     round(float(self._centers[i][j]), 4),
                    'sigma': round(float(self._sigmas[i][j]),  4),
                }
                for j, lbl in enumerate(labels)
            }
            for i in range(4)
        }


# =============================================================================
# NEURAL NETWORK MODEL  (MLP Regressor)
#   Architecture : 4 → 64 → 32 → 1
#   Activation   : ReLU
#   Optimiser    : Adam + early stopping
#   Training     : Backpropagation (sklearn implementation)
# =============================================================================

class NNModel:
    """
    Multilayer Perceptron for AQI regression.
    Trained with backpropagation via the Adam optimiser.
    """

    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            learning_rate_init=0.002,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.architecture = {
            'input_size':    4,
            'hidden_layers': [256, 128, 64],
            'output_size':   1,
            'activation':    'ReLU',
            'optimizer':     'Adam',
        }
        self.train_rmse = None
        self.val_rmse   = None
        self.r2         = None

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        y_p_tr = self.model.predict(X_train)
        y_p_va = self.model.predict(X_val)
        self.train_rmse = float(np.sqrt(mean_squared_error(y_train, y_p_tr)))
        self.val_rmse   = float(np.sqrt(mean_squared_error(y_val,   y_p_va)))
        self.r2         = float(r2_score(y_val, y_p_va))
        return {'train_rmse': self.train_rmse, 'val_rmse': self.val_rmse, 'r2': self.r2}

    def predict(self, x):
        return float(self.model.predict([x])[0])

    def get_info(self):
        return {**self.architecture,
                'train_rmse': self.train_rmse,
                'val_rmse':   self.val_rmse,
                'r2':         self.r2}


# =============================================================================
# ANFIS — Adaptive Neuro-Fuzzy Inference System
#
#   5-layer architecture (Jang 1993):
#     L1  Fuzzification  — Gaussian MFs, learnable (c, σ)
#     L2  Rule strength  — product T-norm
#     L3  Normalisation  — firing strengths / total
#     L4  Consequent     — linear Sugeno:  fₖ = p₀ᵏ + p₁ᵏx₁ + … + pₙᵏxₙ
#     L5  Output         — weighted sum:   y = Σ w̄ₖ fₖ
#
#   Hybrid learning algorithm:
#     Forward  : LSE solves consequent params (p) analytically
#     Backward : Gradient descent updates premise params (c, σ)
# =============================================================================

class ANFIS:
    """
    ANFIS with hybrid learning (LSE + gradient descent).

    Parameters
    ----------
    n_inputs : int   number of input features (default 4)
    n_rules  : int   number of fuzzy rules    (default 6)
    """

    def __init__(self, n_inputs=4, n_rules=6):
        self.n_inputs = n_inputs
        self.n_rules  = n_rules

        rng = np.random.default_rng(42)

        # Premise parameters — centers spread evenly, small random offsets
        self.centers = np.tile(
            np.linspace(0.1, 0.9, n_rules)[:, None], (1, n_inputs)
        ) + rng.uniform(-0.05, 0.05, (n_rules, n_inputs))

        self.sigmas = np.ones((n_rules, n_inputs)) * 0.3

        # Consequent parameters  [p₁, p₂, …, pₙ, bias]  per rule
        self.params = np.zeros((n_rules, n_inputs + 1))

        self.train_rmse      = None
        self.r2              = None
        self._last_activations = None

    # ── Layer implementations ──────────────────────────────────────────────

    def _l1_fuzzify(self, X):
        """L1: Gaussian MFs.  (N, n_inputs) → (N, n_rules, n_inputs)"""
        return np.exp(
            -((X[:, None, :] - self.centers) ** 2) / (2 * self.sigmas ** 2)
        )

    def _l2_fire(self, mu):
        """L2: Product T-norm.  (N, n_rules, n_inputs) → (N, n_rules)"""
        return np.prod(mu, axis=2)

    def _l3_normalise(self, w):
        """L3: Normalised firing strengths."""
        s = np.sum(w, axis=1, keepdims=True)
        return w / np.where(s == 0, 1e-10, s)

    def _l4l5_output(self, X, w_bar):
        """L4+L5: Sugeno consequent + weighted sum."""
        X_aug = np.hstack([X, np.ones((len(X), 1))])   # (N, n+1)
        f     = X_aug @ self.params.T                   # (N, n_rules)
        return np.sum(w_bar * f, axis=1)                # (N,)

    def _forward_batch(self, X):
        mu    = self._l1_fuzzify(X)
        w     = self._l2_fire(mu)
        w_bar = self._l3_normalise(w)
        y_hat = self._l4l5_output(X, w_bar)
        return mu, w, w_bar, y_hat

    # ── Hybrid training ────────────────────────────────────────────────────

    def train(self, X, y, epochs=60, lr=0.005):
        """
        Hybrid learning:
          0. KMeans clustering for premise (rule centers) initialization
          1. Forward pass  → LSE solves consequent params analytically
          2. Backward pass → gradient descent updates premise params (c, σ)
        """
        from sklearn.cluster import KMeans
        N = len(X)
        
        # Use KMeans to set the initial centers to actual dense data regions
        print(f"  [ANFIS] Running KMeans clustering for {self.n_rules} rules...")
        kmeans = KMeans(n_clusters=self.n_rules, random_state=42, n_init='auto')
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        for epoch in range(epochs):
            mu, w, w_bar, _ = self._forward_batch(X)

            # ── Step 1: LSE for consequent parameters ──────────────────
            X_aug = np.hstack([X, np.ones((N, 1))])          # (N, n+1)
            Phi   = (w_bar[:, :, None] * X_aug[:, None, :]).reshape(N, -1)
            theta, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
            self.params = theta.reshape(self.n_rules, self.n_inputs + 1)

            # ── Step 2: GD for premise parameters ──────────────────────
            y_hat  = self._l4l5_output(X, w_bar)
            err    = y - y_hat                              # (N,)

            X_aug2 = np.hstack([X, np.ones((N, 1))])
            f_k    = X_aug2 @ self.params.T                # (N, n_rules)

            w_sum  = np.sum(w, axis=1, keepdims=True)
            d_w    = ((f_k - y_hat[:, None])
                      / np.where(w_sum == 0, 1e-10, w_sum)
                      * err[:, None])                       # (N, n_rules)

            mu_safe = np.where(mu == 0, 1e-10, mu)
            d_mu    = d_w[:, :, None] * w[:, :, None] / mu_safe  # (N, n_rules, n_input)

            diff = X[:, None, :] - self.centers             # (N, n_rules, n_inputs)
            d_c  = d_mu * mu * diff      / (self.sigmas ** 2)
            d_s  = d_mu * mu * diff ** 2 / (self.sigmas ** 3)

            self.centers += lr * np.mean(d_c, axis=0)
            self.sigmas  += lr * np.mean(d_s, axis=0)
            self.sigmas   = np.clip(self.sigmas, 0.01, 1.0)

            if epoch % 10 == 0:
                mse = float(np.mean(err ** 2))
                print(f"  ANFIS epoch {epoch:3d}/{epochs}  MSE={mse:.6f}")

        # Final metrics
        _, _, _, y_final = self._forward_batch(X)
        self.train_rmse = float(np.sqrt(mean_squared_error(y, y_final)))
        self.r2         = float(r2_score(y, y_final))

    # ── Inference ──────────────────────────────────────────────────────────

    def forward(self, x):
        """Single-sample prediction.  x: (n_inputs,)"""
        X = x[np.newaxis, :]
        mu, w, w_bar, y_hat = self._forward_batch(X)
        self._last_activations = {
            'memberships':      mu[0].tolist(),
            'firing_strengths': w[0].tolist(),
            'normalized':       w_bar[0].tolist(),
        }
        return float(y_hat[0])

    def get_rule_activations(self):
        return self._last_activations

    def get_info(self):
        return {
            'n_inputs':   self.n_inputs,
            'n_rules':    self.n_rules,
            'train_rmse': self.train_rmse,
            'r2':         self.r2,
            'centers':    self.centers.tolist(),
            'sigmas':     self.sigmas.tolist(),
        }