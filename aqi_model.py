"""
AirSense — AQI Category Prediction Prototype
----------------------------------------------
This script demonstrates the "prediction" component of the AirSense project.

Since live government AQI APIs aren't reachable from this sandboxed environment,
we generate a REALISTIC SYNTHETIC dataset that follows the actual CPCB/WHO AQI
breakpoint structure (PM2.5, PM10, NO2, O3 -> AQI category), then train a small
classifier on it. In the real deployment, this synthetic generator would be
replaced by a live feed (e.g., CPCB / OpenAQ / IQAir API) — the model logic
and pipeline stay identical.

AQI Categories used (Indian CPCB standard):
  Good (0-50), Satisfactory (51-100), Moderate (101-200),
  Poor (201-300), Very Poor (301-400), Severe (401-500)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------
# 1. Generate a realistic synthetic AQI dataset
# ---------------------------------------------------------
N = 1500

def categorize(pm25):
    if pm25 <= 30: return "Good"
    elif pm25 <= 60: return "Satisfactory"
    elif pm25 <= 90: return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else: return "Severe"

# Simulate a city with seasonal + daily variation (winter = worse air, common in Indian cities)
day_of_year = np.random.randint(0, 365, N)
season_factor = 40 * np.cos((day_of_year / 365) * 2 * np.pi) + 60  # peaks in winter
traffic_factor = np.random.normal(30, 15, N).clip(0, 100)
humidity = np.random.normal(60, 15, N).clip(10, 100)
wind_speed = np.random.normal(8, 4, N).clip(0.5, 30)

pm25 = (season_factor + traffic_factor * 0.8 - wind_speed * 2 + np.random.normal(0, 10, N)).clip(5, 450)
pm10 = pm25 * np.random.uniform(1.3, 1.9, N)
no2 = (traffic_factor * 0.9 + np.random.normal(20, 8, N)).clip(5, 150)
o3 = (np.random.normal(40, 15, N) + (100 - humidity) * 0.2).clip(5, 180)

df = pd.DataFrame({
    "pm25": pm25, "pm10": pm10, "no2": no2, "o3": o3,
    "humidity": humidity, "wind_speed": wind_speed, "traffic_index": traffic_factor
})
df["aqi_category"] = df["pm25"].apply(categorize)

# ---------------------------------------------------------
# 2. Train a classifier to predict AQI category from sensor inputs
# ---------------------------------------------------------
features = ["pm10", "no2", "o3", "humidity", "wind_speed", "traffic_index"]
X = df[features]
le = LabelEncoder()
y = le.fit_transform(df["aqi_category"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

print(f"Model accuracy: {acc:.2%}\n")
print(report)

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature importance (what drives AQI predictions):")
print(importances)

# ---------------------------------------------------------
# 3. Save a chart: 30-day AQI trend for the advisory demo
# ---------------------------------------------------------
trend_days = 30
trend_pm25 = (60 + 40 * np.cos(np.linspace(0, 3, trend_days)) + np.random.normal(0, 12, trend_days)).clip(10, 300)
trend_dates = pd.date_range("2026-07-20", periods=trend_days)

fig, ax = plt.subplots(figsize=(9, 4.2))
colors = []
for v in trend_pm25:
    if v <= 30: colors.append("#2e7d32")
    elif v <= 60: colors.append("#9ccc65")
    elif v <= 90: colors.append("#fdd835")
    elif v <= 120: colors.append("#fb8c00")
    elif v <= 250: colors.append("#e53935")
    else: colors.append("#6a1b9a")

ax.bar(trend_dates, trend_pm25, color=colors, width=0.8)
ax.set_ylabel("PM2.5 (µg/m³)", fontsize=11)
ax.set_title("Kolkata — Simulated 30-Day PM2.5 Trend (Demo Data)", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("/home/claude/airsense/aqi_trend.png", dpi=200)
print("\nSaved trend chart to aqi_trend.png")

# Save the day used for the advisory demo (last day of trend)
latest_pm25 = trend_pm25[-1]
latest_category = categorize(latest_pm25)
print(f"\nLatest simulated reading: PM2.5={latest_pm25:.1f} -> Category: {latest_category}")

df.to_csv("/home/claude/airsense/synthetic_aqi_dataset.csv", index=False)
