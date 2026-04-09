# Air Quality Index Prediction Using Soft Computing Techniques
### A Comparative Study of Fuzzy Logic, Neural Networks, and ANFIS

---

**Course:** Soft Computing  
**Project Title:** AQI Prediction Dashboard — Fuzzy Logic, Neural Network & ANFIS  
**Repository:** https://github.com/Prasad0224/AQI-Prediction-SC

---

## Abstract

Air Quality Index (AQI) prediction is a critical public health problem that requires modelling complex, non-linear relationships between multiple atmospheric pollutants. This report presents the design, implementation, and comparative evaluation of three soft computing paradigms — **Mamdani Fuzzy Inference System**, **Multilayer Perceptron Neural Network (MLP)**, and **Adaptive Neuro-Fuzzy Inference System (ANFIS)** — applied to real-world AQI data. Each method is studied from first principles, implemented from scratch in Python, and evaluated on a common held-out test set. Results demonstrate that after rigorous physical bound filtering and mathematical optimization (like K-Means rule clustering for ANFIS and 5-tier linguistic expansion for Fuzzy logic), the Neural Network achieves the lowest RMSE (~31.19), closely followed by K-Means clustered ANFIS (~37.00). An interactive web dashboard was built to visualise all three models simultaneously.

---

## 1. Introduction

The Air Quality Index is a composite measure quantifying how polluted the air is, computed from concentrations of key pollutants including Particulate Matter (PM2.5, PM10), Nitrogen Dioxide (NO₂), and Carbon Monoxide (CO). Traditional deterministic approaches to AQI prediction (linear regression, lookup tables) fail to capture the uncertainty and non-linearity inherent in atmospheric chemistry.

Soft computing offers three complementary solutions:

- **Fuzzy Logic** encodes expert knowledge as IF-THEN rules, tolerating imprecision through linguistic variables.
- **Neural Networks** learn patterns purely from data through backpropagation, requiring no domain knowledge.
- **ANFIS** combines both — it uses the structural interpretability of fuzzy systems with the learning ability of neural networks.

This study implements all three from first principles on a dataset of ~25,000 hourly AQI readings from multiple Indian cities.

---

## 2. Dataset

| Property | Value |
|---|---|
| **Source** | Indian Air Quality dataset (Kaggle) |
| **Samples** | ~25,000 hourly readings |
| **Input Features** | PM2.5, PM10, NO₂, CO |
| **Target Variable** | AQI (Air Quality Index) |
| **Train / Test Split** | 80% / 20% |
| **Preprocessing** | Min-Max normalisation to [0, 1] via `sklearn.MinMaxScaler` |

Missing values were dropped. The target (AQI) was independently normalised using its own scaler so denormalisation back to real AQI units is lossless.

---

## 3. Soft Computing Methods

### 3.1 Fuzzy Logic — Mamdani Inference System

#### 3.1.1 Theoretical Background

Fuzzy Logic, introduced by Zadeh (1965), extends classical Boolean logic to handle degrees of truth. Instead of a variable being simply "high" or "low", it can simultaneously belong to multiple fuzzy sets with degrees of membership µ ∈ [0, 1].

A **Mamdani Fuzzy Inference System** consists of four stages:

1. **Fuzzification** — maps crisp input values to membership degrees in linguistic sets (Low, Medium, High) via Membership Functions (MFs).
2. **Rule Evaluation** — evaluates a set of IF-THEN rules using the fuzzified inputs.
3. **Aggregation** — combines all rule outputs.
4. **Defuzzification** — converts the aggregated fuzzy output to a crisp value.

#### 3.1.2 Membership Functions

Gaussian MFs are used for smooth, continuously differentiable boundaries:

```
µ(x; c, σ) = exp(−(x − c)² / 2σ²)
```

Where `c` is the centre and `σ` controls the width. **Five linguistic levels** are defined per pollutant mirroring the official Indian standard: **Good**, **Satisfactory**, **Moderate**, **Poor**, **Severe**.

**Calibration from Data:** Rather than statistically blindly assigning clusters, they are computed deeply from the `20th, 40th, 60th, and 80th` physical percentiles of the highly-filtered training data to prevent drift.

This ensures the MF boundaries reflect the actual distribution of each pollutant in the dataset.

#### 3.1.3 Inference Design

The system independently infers a **sub-AQI** for each of the 4 pollutants using centroid defuzzification:

```
sub_AQI_i = Σ (µ_k × AQI_singleton_k) / Σ µ_k
```

Where AQI singletons are calibrated to the centre of real Indian AQI bands:

| Linguistic Level | AQI Singleton | AQI Band |
|---|---|---|
| Low | 30 | Good (0–50) |
| Medium | 100 | Satisfactory / Moderate |
| High | 220 | Unhealthy / Very Unhealthy |

**Final AQI = max(sub-AQI_PM2.5, sub-AQI_PM10, sub-AQI_NO₂, sub-AQI_CO)**

This mirrors how real-world AQI is calculated: the dominant (worst) pollutant determines the overall index.

#### 3.1.4 Key Design Decisions

- All 4 pollutants used (previous 2-input version caused systematic underestimation).
- MF calibration from data percentiles instead of hardcoded values eliminates bias.
- The max-aggregation rule matches the official Indian AQI standard.

---

### 3.2 Neural Network — Multilayer Perceptron (MLP)

#### 3.2.1 Theoretical Background

An MLP is a feedforward network of layers of artificial neurons. Each neuron computes a weighted sum of its inputs, applies a non-linear activation function, and passes the result to the next layer. Learning occurs through **backpropagation** — the chain rule of calculus applied recursively from output to input layers to compute gradients of the loss function with respect to all weights.

#### 3.2.2 Architecture

```
Input (4) → Hidden₁ (64, ReLU) → Hidden₂ (32, ReLU) → Output (1, linear)
```

| Component | Value |
|---|---|
| **Input size** | 4 (PM2.5, PM10, NO₂, CO) |
| **Hidden layers** | [256, 128, 64] neurons |
| **Activation** | ReLU: f(x) = max(0, x) |
| **Output** | 1 neuron, linear (regression) |
| **Loss function** | Mean Squared Error (MSE) |
| **Optimiser** | Adam (Adaptive Moment Estimation) |
| **Regularisation** | Early stopping |

#### 3.2.3 Training Process — Backpropagation

For each training batch:

1. **Forward pass** — compute prediction ŷ through all layers.
2. **Loss** — MSE: L = (1/N) Σ (yᵢ − ŷᵢ)²
3. **Backward pass** — compute ∂L/∂W for all weights using chain rule.
4. **Adam update** — adaptive learning rate per parameter using first and second moment estimates:
   - m_t = β₁ m_{t-1} + (1−β₁) g_t  
   - v_t = β₂ v_{t-1} + (1−β₂) g_t²  
   - W_t = W_{t-1} − α m̂_t / (√v̂_t + ε)

#### 3.2.4 ReLU Activation

ReLU is used in hidden layers because it:
- Avoids vanishing gradients (constant gradient for positive inputs)
- Introduces sparsity (neurons can be exactly zero)
- Is computationally cheap

The linear output neuron allows the network to predict any real-valued AQI.

---

### 3.3 ANFIS — Adaptive Neuro-Fuzzy Inference System

#### 3.3.1 Theoretical Background

ANFIS (Jang, 1993) implements a Sugeno-type fuzzy system as a 5-layer neural network. The architecture allows both the premise parameters (MF shapes) and consequent parameters (rule outputs) to be learned from data — combining the interpretability of fuzzy rules with the data-driven learning of neural networks.

#### 3.3.2 Five-Layer Architecture

**Layer 1 — Fuzzification (Gaussian MFs):**
```
O¹ᵢ = µᵢ(x) = exp(−(x − cᵢ)² / 2σᵢ²)
```
Parameters `c` (centre) and `σ` (width) are learnable premise parameters.

**Layer 2 — Rule Firing Strength (Product T-norm):**
```
wₖ = Π µᵢₖ(xᵢ)    for rule k
```
Product of all input membership degrees for rule k.

**Layer 3 — Normalised Firing Strength:**
```
w̄ₖ = wₖ / Σⱼ wⱼ
```
Ensures all rule contributions sum to 1.

**Layer 4 — Consequent (Sugeno Linear):**
```
fₖ = p₀ₖ + p₁ₖ x₁ + p₂ₖ x₂ + ... + pₙₖ xₙ
```
Each rule has its own linear function of inputs. Parameters `p` are consequent parameters.

**Layer 5 — Output (Weighted Sum):**
```
ŷ = Σₖ w̄ₖ fₖ
```

#### 3.3.3 Hybrid Learning Algorithm

ANFIS uses a hybrid two-phase learning approach per epoch:

**Phase 1 — Forward pass, Least Squares Estimation (LSE):**
Given fixed premise parameters, the output ŷ is linear in the consequent parameters `p`. Therefore the optimal `p` can be solved analytically in one step using the Moore-Penrose pseudoinverse — far faster and more stable than gradient descent for these parameters.

```
Φ · p = y  →  p* = (ΦᵀΦ)⁻¹ Φᵀ y
```

Where Φ is the regressor matrix of (w̄ₖ × xᵢ) terms.

**Phase 2 — Backward pass, Gradient Descent:**
With consequent parameters fixed at p*, gradients are propagated backward to update premise parameters (c, σ) via standard gradient descent.

This hybrid approach converges significantly faster than pure gradient descent on all parameters simultaneously.

#### 3.3.4 Implementation Details

| Parameter | Value |
|---|---|
| **Number of rules** | 6 |
| **Number of inputs** | 4 |
| **Epochs** | 60 |
| **Learning rate** | 0.005 |
| **Initial centres** | KMeans clustering using `sklearn.cluster` |

---

## 4. Implementation

All three models are implemented in Python in `backend/model.py`. Training is orchestrated by `backend/train.py`, which:

1. Loads the dataset via `backend/preprocess.py`
2. Splits into 80/20 train/test
3. Fits Min-Max scalers on training data only
4. Trains NN and ANFIS on normalised data, Fuzzy calibrated from percentiles
5. Evaluates all models on the held-out test set in real AQI units
6. Serialises all three models + scalers to `models.pkl` via `pickle`

The REST API (`backend/app.py`, Flask) exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Returns Fuzzy, NN, ANFIS predictions + per-pollutant explanation |
| `/model-info` | GET | Returns architecture details, MF parameters, rule activations |
| `/live` | GET | Fetches real-time AQI from WAQI API |
| `/health` | GET | Server health check |

The React frontend (`frontend/src/`) provides dedicated panels for each model with mathematical visualisations, MF charts, and rule activation displays.

---

## 5. Results

### 5.1 Performance Metrics

All metrics computed on the **held-out 20% test set** in **real AQI units** (not normalised).

| Model | Test RMSE (AQI units) | R² Score |
|---|---|---|
| **Neural Network (MLP)** | **~31.19** | **~0.89** |
| **ANFIS (K-Means Clustered)** | ~37.00 | ~0.85 |
| **Fuzzy Logic (Mamdani 5-Tier)** | ~76.20 | ~0.39 |

### 5.2 Interpretation

**Neural Network** achieves the best performance (RMSE = 31.19) because it utilizes an expanded `256+128+64 = 448` neuron capacity to overapproximate the sharp `max()` piece-wise calculations typical of Indian AQI.

**ANFIS** drastically improved (RMSE = 37.00) due to utilizing `K-Means clustering` to naturally place premise parameters exactly on the dense mathematical clouds of the air-quality inputs rather than uniform guessing.

**Fuzzy Logic** remains highly interpretable, and because its parameters were pushed to 5-tier bounds (`Good ➔ Severe`), it naturally clusters outputs much closer to reality than previous iterations.

### 5.3 Sample Predictions

Input: PM2.5 = 50, PM10 = 80, NO₂ = 20, CO = 1.2

| Model | AQI Prediction | Category |
|---|---|---|
| Fuzzy Logic | ~110 | Unhealthy for Sensitive Groups |
| Neural Network | ~104 | Moderate |
| ANFIS | ~121 | Unhealthy for Sensitive Groups |

Input: PM2.5 = 80, PM10 = 120, NO₂ = 40, CO = 2.0

| Model | AQI Prediction | Category |
|---|---|---|
| Fuzzy Logic | ~141 | Unhealthy for Sensitive Groups |
| Neural Network | ~185 | Unhealthy |
| ANFIS | ~188 | Unhealthy |

All three models agree on the AQI band for moderate-to-high pollution inputs, demonstrating consistency.

---

## 6. Comparative Analysis

### 6.1 Method Comparison

| Criterion | Fuzzy Logic | Neural Network | ANFIS |
|---|---|---|---|
| **Training required** | No (rule-based) | Yes (backprop) | Yes (hybrid LSE + GD) |
| **Interpretability** | ✅ High — rules are human-readable | ❌ Black box | ✅ Medium — rules visible, weights learned |
| **Data requirement** | Low | High | Medium |
| **Accuracy (this study)** | Moderate | ✅ Best | Good |
| **Handles uncertainty** | ✅ Native (linguistic variables) | ❌ Indirect | ✅ Via MFs |
| **Generalisation** | Limited (fixed rules) | Strong | Strong |
| **Computational cost** | ✅ Very low | Medium | Medium |
| **Expert knowledge needed** | ✅ Yes | ❌ No | Partial |

### 6.2 When to Use Each

**Fuzzy Logic** is preferred when:
- Domain expert knowledge is available and reliable
- Interpretability is mandatory (e.g., regulatory or medical systems)
- Training data is scarce
- Real-time inference is needed on embedded hardware

**Neural Networks** are preferred when:
- Large labelled datasets are available
- Maximum accuracy is the priority
- Interpretability is not required
- The relationship between inputs and outputs is highly complex

**ANFIS** is preferred when:
- Both accuracy AND interpretability are needed
- Some domain structure (fuzzy partitioning) is known
- Training data is moderate in size
- The system needs to adapt as data grows (online learning possible)

### 6.3 Limitations

| Model | Limitation |
|---|---|
| Fuzzy | R² = 0.34 — limited accuracy due to fixed rules; rule base does not scale to high dimensions |
| NN | Black-box; no inherent uncertainty quantification; needs significant data |
| ANFIS | Rule explosion with too many inputs (curse of dimensionality); assumptions of Sugeno linearity |

---

## 7. Conclusion

This project demonstrates the design, implementation, and comparative evaluation of three foundational soft computing techniques for the AQI prediction problem:

1. **Fuzzy Logic** provides a transparent, interpretable baseline with no training, achieving R² = 0.338 after calibrating MF centres from training-data percentiles.
2. **Neural Network (MLP)** achieves the highest accuracy (R² = 0.891, RMSE = 44.73) by learning purely from data using backpropagation.
3. **ANFIS** bridges the gap (R² = 0.776, RMSE = 64.04) by combining fuzzy structure with hybrid LSE + gradient descent learning.

The complementary strengths of these three approaches reflect the core principle of soft computing: no single method is optimal for all settings. The interactive dashboard makes these trade-offs tangible — users can run all three models simultaneously, inspect MF curves, rule activations, and layer-by-layer ANFIS outputs.

---

## References

1. Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
2. Jang, J.S.R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
3. Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
4. Kingma, D.P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.
5. Central Pollution Control Board (CPCB). Indian Air Quality Index — Calculation Methodology.

---

*End of Report*
