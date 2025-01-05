# Bitcoin Price Prediction using Machine Learning

This project focuses on predicting Bitcoin prices using advanced Machine Learning techniques, with a primary focus on LSTM (Long Short-Term Memory) models for time series forecasting. The project also includes Exploratory Data Analysis (EDA) and a user-friendly Streamlit application for visualizing predictions and analytics.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)
6. [Results](#results)
7. [Streamlit Dashboard](#streamlit-dashboard)
8. [License](#license)

---

## Project Overview
Bitcoin, as the leading cryptocurrency, exhibits significant price volatility. Predicting its price trends can benefit traders, investors, and researchers. This project:
- Analyzes historical Bitcoin price data from 2012 to 2024.
- Conducts thorough Exploratory Data Analysis (EDA) to uncover trends and insights.
- Compares various machine learning models for time series forecasting, including LSTM.
- Provides an interactive Streamlit dashboard for visualizing predictions and insights.

---

## Dataset
The dataset used contains minute-by-minute Bitcoin price data from January 1, 2012, to December 12, 2024. It includes key features such as Open, High, Low, Close prices, and trading volume.

You can download the dataset from the following Google Drive link:
[Download the Dataset](<https://drive.google.com/file/d/1Aeqv_V9VV5_iTRglWc7knW1GGRLy46YC/view?usp=sharing>)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AlejandroMRZ/An-ML-approach-to-historical-Bitcoin-s-price.git
   cd An-ML-approach-to-historical-Bitcoin-s-price
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Git LFS (if needed for large files):
   ```bash
   git lfs install
   git lfs pull
   ```

---

## Usage
1. **Run Exploratory Data Analysis (EDA):**
   - Open the `BTC_historical_analysis.ipynb` notebook to explore the data.

2. **Train the Model:**
   - Use the `BTC_price_forecasting_using_LSTM.ipynb` notebook to train and evaluate models.

3. **Launch the Streamlit Dashboard:**
   - Run the Streamlit app for predictions and analytics:
     ```bash
     streamlit run btc_app.py
     ```

---

## Features
- **Exploratory Data Analysis:**
  - Trend analysis, seasonality detection, and correlation analysis.
- **Machine Learning Models:**
  - Baseline models (e.g., Linear Regression).
  - Advanced models (e.g., LSTM for time series).
- **Interactive Dashboard:**
  - Real-time visualization of Bitcoin prices and predictions.

---

## Results
The LSTM model outperformed baseline models in capturing the non-linear and volatile nature of Bitcoin price movements. Key metrics (e.g., RMSE, MAE) demonstrate its predictive accuracy.

---

## Streamlit Dashboard
The Streamlit dashboard provides:
1. **Interactive Visualizations:**
   - Historical price trends.
   - Predicted vs. actual prices.
2. **Real-Time Predictions:**
   - Upload new data for instant predictions.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

For any questions or issues, feel free to open an issue in this repository or contact me directly!

