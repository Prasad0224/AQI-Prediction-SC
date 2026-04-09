# 🌍 AQI Prediction — Soft Computing Dashboard

> An interactive, educational web application demonstrating **Fuzzy Logic**, **Neural Networks (MLP)**, and **ANFIS (Adaptive Neuro-Fuzzy Inference System)** applied to real-world Air Quality Index prediction.
>
> Built as a college mini-project to showcase core **Soft Computing** concepts with a full-stack implementation (Flask + React).

---

## 📸 Dashboard Preview

- **3 Soft Computing Models:** Compare **Mamdani Fuzzy Inference**, a **Deep Neural Network (MLP)**, and a **K-Means Clustered ANFIS** on the same inputs.
- **Explainable Web Dashboard:** Built with React & Flask. Visualise Fuzzy Membership Functions, live Rule Activations, and human-readable reasoning dynamically.
- **Live Data:** Fetch real-time pollutant levels from the World Air Quality Index (WAQI) API.

| Neural Network Tab | ANFIS Tab |
|---|---|
| SVG layer diagram + backprop steps | 5-layer architecture + firing strengths |

---

## 🧠 Soft Computing Models

### 1. 🔵 Fuzzy Logic (Mamdani FIS)
- **Design:** Expert-defined 5-tier fuzzy rules matching the Indian CPCB system (Good → Severe).
- **Membership Functions:** Gaussian MFs dynamically calibrated to the 20th, 40th, 60th, and 80th percentiles of the training strict data.
- **Inference:** Each pollutant computes a sub-AQI via centroid defuzzification. Final AQI is the `max()` of all sub-AQIs, perfectly mirroring the official mathematical standard.

### 2. 🟣 Neural Network (MLP)
- **Architecture**: `4 → 256 → 128 → 64 → 1`
- **Why**: The robust architecture over-approximates the sharp, non-linear piece-wise boundaries of true AQI calculation.
- **Details:** ReLU activation, Adam optimiser, trained purely on observed data with backpropagation.

### 3. 🟠 ANFIS (Adaptive Neuro-Fuzzy Inference System)
- **Design:** 5-layer network combining Fuzzy structural interpretability with NN data-driven learning.
- **Initialization:** Features **K-Means clustering** to intelligently place rules onto dense mathematical physical data distributions instead of random guesses.
- **Hybrid Learning**:
  - **Forward pass**: LSE solves Sugeno consequent params (`f = px + qy + r`) analytically
  - **Backward pass**: Gradient descent updates Gaussian MF centers (c) and widths (σ)

---

## 📁 Project Structure

```
AQI-Prediction/
├── backend/
│   ├── app.py              # Flask REST API (6 endpoints)
│   ├── model.py            # FuzzyModel, NNModel, ANFIS classes
│   ├── train.py            # Training pipeline with metrics
│   ├── preprocess.py       # Data cleaning & feature extraction
│   ├── live_data.py        # Real-time WAQI API integration
│   ├── .env                # API key (not in git — see .env.example)
│   └── .env.example        # Template for environment variables
├── data/
│   └── city_day.csv        # Daily AQI data — model training (Kaggle)
├── frontend/
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js          # Main app — sidebar navigation, global state
│       ├── index.css       # Dark-mode design system
│       └── components/
│           ├── InputForm.js        # Pollutant inputs + live data fetch
│           ├── AQIGauge.js         # Canvas semicircular AQI gauge
│           ├── HealthAdvisory.js   # Health tips based on AQI level
│           ├── ModelComparison.js  # Bar chart — 3 models side by side
│           ├── FuzzyPanel.js       # MF curves, rule table, activations
│           ├── NNPanel.js          # SVG architecture + training metrics
│           └── ANFISPanel.js       # 5-layer diagram + rule firing chart
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Dataset from Kaggle: [India Air Quality Data](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
  - Place `city_day.csv` in the `data/` folder

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AQI-Prediction.git
cd AQI-Prediction
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure API key
cp backend/.env.example backend/.env
# Edit backend/.env and add your WAQI API key
# Get a free key at: https://aqicn.org/data-platform/token/

# Train all three models (generates models.pkl)
cd backend
python train.py
```

**Training output example:**
```
[2/5] Training Neural Network (MLP 4→64→32→1)...
      Train RMSE : 5.24   Test RMSE : 6.98   R²: 0.8908

[3/5] Training ANFIS (hybrid learning: LSE + GD)...
      ANFIS epoch   0/60  MSE=0.000589
      Test  RMSE : 9.14   R²: 0.7761

[4/5] Evaluating Fuzzy Logic (Mamdani, 9 rules)...
      Test  RMSE : 45.21 (real AQI units)  R²: 0.312

[5/5] Results Summary  (all RMSE in real AQI units)
  Model         Test RMSE        R²
  Neural Net        6.98      0.8908
  ANFIS             9.14      0.7761
  Fuzzy Logic      45.21      0.3120
```

### 3. Start the Backend
```bash
# From the backend/ directory
python app.py
# → Flask running on http://localhost:5000
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm start
# → React running on http://localhost:3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Run all 3 models on given pollutant values |
| `POST` | `/explain/fuzzy` | Fuzzy MF memberships + rule activations |
| `POST` | `/explain/anfis` | ANFIS per-rule firing strengths |
| `GET`  | `/model-info` | Architecture metadata for all 3 models |
| `GET`  | `/live/<city>` | Real-time AQI from WAQI API |

**Example `/predict` request:**
```json
{
  "PM2.5": 85,
  "PM10": 120,
  "NO2": 45,
  "CO": 2.1
}
```

**Response:**
```json
{
  "ANFIS": 142.5,
  "NN": 138.2,
  "Fuzzy": 125.0,
  "explanation": {
    "PM2.5": "High",
    "PM10":  "Medium",
    "NO2":   "Low",
    "CO":    "Low"
  }
}
```

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| **AQI Gauge** | Animated canvas semicircle, colour-coded Good→Hazardous |
| **3-Model Comparison** | Bar chart showing Fuzzy, NN, ANFIS side by side |
| **Fuzzy Explainer** | Live Gaussian MF curves with current-input marker + rule table |
| **NN Explainer** | SVG network diagram + training metrics (RMSE, R²) |
| **ANFIS Explainer** | 5-layer architecture diagram + rule firing strength chart |
| **Live Data** | One-click fill from real WAQI sensor data |
| **Health Advisory** | Category-specific health tips and pollutant level badges |

---

## 🗂️ Dataset

**Source**: [India Air Quality Data — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

| File | Used For | Size |
|---|---|---|
| `city_day.csv` | Model training (PM2.5, PM10, NO2, CO → AQI) | ~2.5 MB |

> **Note**: This file is excluded from the repository due to size. Download from Kaggle and place in `data/`.

---

## 🎯 Key Concepts Demonstrated

| Concept | Where |
|---|---|
| Gaussian Membership Functions | `model.py` → `FuzzyModel` + `FuzzyPanel.js` |
| Mamdani Rule Base | `model.py` → `RULES` + rule table in `FuzzyPanel.js` |
| Centroid Defuzzification | `model.py` → `FuzzyModel.predict()` |
| Backpropagation | `model.py` → `NNModel` + `NNPanel.js` |
| Hybrid Learning (LSE + GD) | `model.py` → `ANFIS.train()` + `ANFISPanel.js` |
| Sugeno-type Consequent | `model.py` → `ANFIS._l4l5_output()` |

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3, Flask, Flask-CORS |
| ML Models | NumPy, scikit-learn (MLP), custom ANFIS |
| Data | Pandas, MinMaxScaler |
| Frontend | React 18, Chart.js, Axios |
| Styling | Vanilla CSS (dark mode, glassmorphism) |
| Live Data | WAQI API |

---