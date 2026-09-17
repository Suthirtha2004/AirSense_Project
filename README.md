# AirSense — AQI Category Prediction Prototype

AirSense is a lightweight machine learning project that predicts air quality categories from environmental sensor inputs such as PM2.5, PM10, NO2, O3, humidity, wind speed, and traffic conditions. The project uses a synthetic dataset that follows realistic AQI breakpoints and trains a Random Forest classifier to classify a location into AQI categories.

## Overview

This project is designed as a prototype for the "prediction" component of an AirSense system. Since live government or public AQI APIs are not available in the current sandbox environment, the script creates a realistic synthetic dataset instead of pulling real-world readings. The data generation logic follows the AQI thresholds used in the Indian CPCB standard, and the trained model uses the same pipeline that would be used with real sensor data in deployment.

## AQI Categories

The model classifies air quality into the following categories based on PM2.5 values:

- Good: 0–50
- Satisfactory: 51–100
- Moderate: 101–200
- Poor: 201–300
- Very Poor: 301–400
- Severe: 401–500

These bands are consistent with the CPCB AQI system used in India.

## Project Files

- `aqi_model.py` — Generates the synthetic AQI dataset, trains the Random Forest model, evaluates the model, and saves a trend chart.
- `synthetic_aqi_dataset.csv` — A generated dataset of simulated air quality observations.
- `README.md` — Project overview and usage instructions.

## Dataset

The synthetic dataset contains 1,500 simulated air-quality observations. Each row includes:

- `pm25`
- `pm10`
- `no2`
- `o3`
- `humidity`
- `wind_speed`
- `traffic_index`
- `aqi_category`

The data generation process simulates realistic seasonal and daily variation, including:

- stronger pollution in winter-like conditions
- traffic-related contribution to NO2 and PM2.5
- effect of humidity and wind speed on pollutant levels
- realistic pollutant ranges for city environments

## Model Pipeline

The model workflow in `aqi_model.py` is:

1. Generate synthetic AQI records.
2. Map PM2.5 values to AQI categories.
3. Select relevant environmental features.
4. Encode the category labels.
5. Split the data into training and testing sets.
6. Train a `RandomForestClassifier`.
7. Evaluate the model using accuracy and a classification report.
8. Print feature importance to show which variables influence predictions most.
9. Save a 30-day PM2.5 trend chart.
10. Export the synthetic dataset to CSV.

## Model Type

The project uses a Random Forest classifier because it performs well on tabular sensor data and provides interpretable feature importance values. The model is trained on engineered synthetic measurements rather than live API data.

## Installation

Make sure Python 3.9+ is installed, then install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

Run the project from the workspace root:

```bash
python aqi_model.py
```

When the script executes, it will:

- print the model accuracy
- display the classification report
- show feature importance rankings
- save a PM2.5 trend chart
- save the synthetic dataset as CSV

## Example Output

The script prints metrics such as:

```text
Model accuracy: 82.33%
```

and a classification report with per-category precision, recall, and F1-score.

## Output Files

The script is designed to save:

- a 30-day AQI trend chart (`aqi_trend.png`)
- a generated CSV dataset (`synthetic_aqi_dataset.csv`)

In the current script, the save paths are set to a Linux-style absolute path, so they may need to be adjusted for a local Windows environment.

## Notes

This repository is intentionally a prototype and a demonstration of the modeling logic. In a production deployment, the synthetic generator would be replaced by a real data source such as:

- CPCB API
- OpenAQ
- IQAir
- a local sensor network

The machine learning pipeline and AQI categorization logic are kept the same, making it easy to swap the data source without redesigning the model.

## Future Improvements

- connect to a real AQI API
- add more pollutant features and geographic data
- compare multiple models such as XGBoost or Gradient Boosting
- add a web dashboard or REST API for AQI forecast and advisory output
- generate hourly or daily AQI predictions for multiple cities

## License

This project is intended for educational and demonstration purposes.
