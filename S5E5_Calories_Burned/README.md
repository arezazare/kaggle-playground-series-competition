# 🔥 S5E5 – Calories Burned Prediction (Kaggle Playground Series)

This project is my submission for **Season 5, Episode 5** of the [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s5e5).  
The goal was to build a regression model to predict **calories burned** during a workout using biometric and activity data.

The competition was evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**.

---

## 📊 Problem Overview

- **Problem Type:** Regression
- **Target Variable:** `Calories`
- **Metric:** RMSLE
- **Data Shape:**  
  - Train: 750,000 rows × 9 columns  
  - Test: 250,000 rows × 8 columns

---


## 🧠 Workflow Summary

### ✅ 1. Data Cleaning
- Removed duplicates, no nulls
- Capped outliers using IQR
- Encoded categorical variable: `Sex`

### ✅ 2. Feature Engineering
Created meaningful features to enhance model signal:
- `BMI = Weight / Height²`
- `Cardio_Load = Heart_Rate × Duration`
- `WeightPerMin`, `EffortPerBMI`, `HR_per_Age`, `Temp_per_Age`
- Applied `log1p()` to right-skewed features

### ✅ 3. Modeling Strategy
- Used `StandardScaler` for scaling
- Transformed target with `log1p()` to align with RMSLE
- Pre-tested 8 models: Ridge, KNN, RandomForest, XGBoost, LightGBM, etc.
- Best results from:
  - **XGBoost (post-tuning)**
  - **LightGBM (pre-tuning)**

### ✅ 4. Tuning
- Used `RandomizedSearchCV` with `n_iter=15`, `cv=3`
- Tuned top 3 models only to reduce compute time
- Best RMSLE from **XGBoost = 0.0617**
- **LightGBM pre-tuning = 0.0613** → better than its tuned version

### ✅ 5. Ensemble & Final Prediction
- Used weighted blend:
  ```python
  final_preds = 0.65 * xgb_preds + 0.35 * lgb_preds
  final_preds = np.expm1(final_preds_log)
  final_preds = np.maximum(final_preds, 0)
  ```

---

## 🏁 Final Results

| Metric           | Result         |
|------------------|----------------|
| **Public RMSLE**  | **0.05928** ✅  
| **Leaderboard Rank** | **1710 / 3205**  
| **Strategy**      | Log-target regression + model blending (XGBoost + LightGBM)  
| **Post-processing** | Clipped extreme predictions to reduce RMSLE spikes  

---

## 🧠 Skills Practiced in This Episode

- 📈 Regression modeling with RMSLE optimization
- 🧮 Feature engineering (ratios like BMI, HR_per_BMI, and interactions)
- 🔄 Log-transforming both the target and skewed inputs
- 🔧 Hyperparameter tuning with `RandomizedSearchCV`
- 🧠 Ensemble modeling via weighted blending
- 🧹 Outlier handling and post-prediction clipping

---

## 📎 Links

- 📄 [Competition Page on Kaggle](https://www.kaggle.com/competitions/playground-series-s5e5)
- 🔙 [Back to Main Repo](../)

---

## 🙋‍♂️ Author

**Reza Zare**  
📌 [GitHub](https://github.com/arezazare) • 🧠 [Kaggle](https://www.kaggle.com/arezazare) • 🌐 [Portfolio](https://arezazare.github.io)

---
