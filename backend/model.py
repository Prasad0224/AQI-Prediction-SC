import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# FUZZY LOGIC MODEL
# Mamdani Fuzzy Inference System
#   - Gaussian membership functions (learnable parameters shown explicitly)
#   - 9-rule base: PM2.5-level × NO2-level → AQI-level
#   - Centroid defuzzification
# =============================================================================

class FuzzyModel:
    """
    Mamdani Fuzzy Inference System for AQI prediction.

    Inputs  : PM2.5 (x[0]), NO2 (x[2])  — both normalised [0, 1]
    Output  : REAL AQI value (not normalised) — singletons are calibrated
              to standard AQI breakpoints so no inverse-transform is needed.

    Membership functions : Gaussian  μ(x) = exp(-(x-c)²/2σ²)
    Rule aggregation     : min (AND)
    Defuzzification      : centroid
    """

    # Flag: output is real AQI, NOT normalised [0,1]
    OUTPUT_IS_NORMALIZED = False

    # --- Premise MFs ---
    PM25_MFS = {
        'Low':    {'c': 0.15, 'sigma': 0.12},
        'Medium': {'c': 0.45, 'sigma': 0.12},
        'High':   {'c': 0.80, 'sigma': 0.12},
    }
    NO2_MFS = {
        'Low':    {'c': 0.15, 'sigma': 0.12},
        'Medium': {'c': 0.45, 'sigma': 0.12},
        'High':   {'c': 0.80, 'sigma': 0.12},
    }

    # --- Consequent singletons: real AQI values aligned to standard categories ---
    # Good 0-50 | Moderate 51-100 | USG 101-150 | Unhealthy 151-200 | Hazardous 201+
    AQI_SINGLETONS = {
        'Good':      25,    # centre of Good band
        'Moderate':  75,    # centre of Moderate band
        'USG':       125,   # centre of USG band
        'Unhealthy': 175,   # centre of Unhealthy band
        'Hazardous': 300,   # representative Hazardous value
    }

    # --- Rule base: (PM2.5, NO2) -> AQI ---
    RULES = [
        ('Low',    'Low',    'Good'),
        ('Low',    'Medium', 'Moderate'),
        ('Low',    'High',   'USG'),
        ('Medium', 'Low',    'Moderate'),
        ('Medium', 'Medium', 'USG'),
        ('Medium', 'High',   'Unhealthy'),
        ('High',   'Low',    'USG'),
        ('High',   'Medium', 'Unhealthy'),
        ('High',   'High',   'Hazardous'),
    ]

    @staticmethod
    def _gaussian(x, c, sigma):
        return float(np.exp(-((x - c) ** 2) / (2 * sigma ** 2)))

    def _memberships(self, x):
        pm25 = float(x[0])
        no2  = float(x[2])
        pm25_mu = {k: self._gaussian(pm25, v['c'], v['sigma']) for k, v in self.PM25_MFS.items()}
        no2_mu  = {k: self._gaussian(no2,  v['c'], v['sigma']) for k, v in self.NO2_MFS.items()}
        return pm25_mu, no2_mu

    def predict(self, x):
        pm25_mu, no2_mu = self._memberships(x)
        num = den = 0.0
        for pm25_lbl, no2_lbl, aqi_lbl in self.RULES:
            strength = min(pm25_mu[pm25_lbl], no2_mu[no2_lbl])
            num += strength * self.AQI_SINGLETONS[aqi_lbl]
            den += strength
        return num / den if den > 1e-9 else 0.5

    def get_explanation(self, x):
        pm25_mu, no2_mu = self._memberships(x)
        rule_activations = []
        for pm25_lbl, no2_lbl, aqi_lbl in self.RULES:
            strength = min(pm25_mu[pm25_lbl], no2_mu[no2_lbl])
            rule_activations.append({
                'pm25_label': pm25_lbl,
                'no2_label':  no2_lbl,
                'aqi_output': aqi_lbl,
                'strength':   round(float(strength), 4),
            })
        return {
            'pm25_memberships': {k: round(v, 4) for k, v in pm25_mu.items()},
            'no2_memberships':  {k: round(v, 4) for k, v in no2_mu.items()},
            'rule_activations': rule_activations,
        }

    def get_mf_params(self):
        return {
            'PM2.5': {k: dict(v) for k, v in self.PM25_MFS.items()},
            'NO2':   {k: dict(v) for k, v in self.NO2_MFS.items()},
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
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.architecture = {
            'input_size':    4,
            'hidden_layers': [64, 32],
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
          1. Forward pass  → LSE solves consequent params analytically
          2. Backward pass → gradient descent updates premise params (c, σ)
        """
        N = len(X)
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