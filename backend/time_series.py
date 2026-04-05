import numpy as np
from sklearn.neural_network import MLPRegressor
from preprocess import clean_city_hour

LAG = 12


class TimeSeriesForecaster:
    """
    MLP-based time-series forecaster using lag features.
    Trained once per city and cached in memory.
    Supports multi-step ahead forecasting.
    """

    def __init__(self):
        self._cache = {}   # city_key -> (model, s_min, s_max, last_window)

    def _load_series(self, city, path):
        df   = clean_city_hour(path)
        mask = df['City'].str.strip().str.lower() == city.lower()
        return df.loc[mask, 'AQI'].dropna().values

    @staticmethod
    def _make_lags(series, lag=LAG):
        X, y = [], []
        for i in range(lag, len(series)):
            X.append(series[i - lag: i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def _train(self, city, path):
        series = self._load_series(city, path)
        if len(series) < LAG + 10:
            return False

        s_min, s_max = series.min(), series.max()
        span = s_max - s_min if s_max != s_min else 1.0
        s_norm = (series - s_min) / span

        X, y = self._make_lags(s_norm, LAG)
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            random_state=42,
        )
        model.fit(X, y)
        self._cache[city.lower()] = (model, s_min, s_max, s_norm)
        print(f"[TS] Forecaster ready for '{city}'")
        return True

    def predict(self, city, steps=24, path="../data/city_hour.csv"):
        key = city.lower()
        if key not in self._cache:
            ok = self._train(city, path)
            if not ok:
                return []

        model, s_min, s_max, s_norm = self._cache[key]
        span = s_max - s_min if s_max != s_min else 1.0

        history = list(s_norm[-LAG:])
        preds_norm = []
        for _ in range(steps):
            x_in = np.array(history[-LAG:]).reshape(1, -1)
            p    = float(model.predict(x_in)[0])
            preds_norm.append(p)
            history.append(p)

        return [round(float(p * span + s_min), 2) for p in preds_norm]