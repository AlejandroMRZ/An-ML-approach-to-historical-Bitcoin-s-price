# Project Memory: Bitcoin Price Prediction Using Machine Learning

## Overview
This project aimed to analyze historical Bitcoin price data and build predictive models using machine learning techniques. Exploratory Data Analysis (EDA) was performed to uncover trends, correlations, and patterns in the data, and a regression-based model was developed to predict Bitcoin prices. Additionally, moving averages were integrated into the model to improve its performance.

---

## Key Insights from Exploratory Data Analysis (EDA)
### Observations:
1. **Perfect Correlation Between Price Metrics**:
   - Features like `Open`, `High`, `Low`, and `Close` prices have a correlation of **1.0** due to their aggregated derivation at the monthly level.
   - These metrics move together, providing consistent data patterns.

2. **Negative Correlation Between `Volume` and Price Metrics**:
   - `Volume` shows a **-0.43** correlation with price metrics.
   - Higher trading volumes tend to be associated with lower prices, suggesting **selling pressure** during high activity.

3. **Moderate Positive Correlation Between `Volume` and `Monthly_Return`**:
   - A **0.28** correlation indicates that higher trading volumes are linked to significant price movements and returns.

4. **Weak Correlation Between `Monthly_Return` and Price Metrics**:
   - A correlation of **-0.052** suggests that monthly returns depend more on price changes than absolute price levels.

---

## Machine Learning Model Development
### Model Trained:
- **Regression Model** for Bitcoin price prediction.

### Evaluation Metrics:
1. **Mean Squared Error (MSE)**: `35,037,661.21`
   - Represents the average squared difference between predicted and actual prices.

2. **Root Mean Squared Error (RMSE)**: `18,718.35`
   - Shows an average deviation of approximately $18,718 from actual BTC prices.

3. **Mean Absolute Error (MAE)**: `14,410.20`
   - Indicates that predictions deviate on average by about $14,410.

4. **Mean Absolute Percentage Error (MAPE)**: `36.15%`
   - Highlights that the model’s predictions vary by a significant percentage from the actual values.

5. **R² Score**: `0.145`
   - Indicates that the model explains 14.5% of the variance in BTC prices, capturing only a small portion of meaningful patterns.

---

## Model Enhancements
### Moving Averages as Features:
1. **50-Day Moving Average (MA)**:
   - Captures short-to-mid-term trends in BTC prices.

2. **200-Day Moving Average (MA)**:
   - Reflects long-term trends.

### Insights:
- Including these moving averages provided the model with momentum and directional bias over meaningful time horizons, helping smooth out short-term price fluctuations.

---

## Challenges and Insights
1. **Volatility of BTC Prices**:
   - The model struggled to account for Bitcoin’s inherent volatility, as highlighted by high RMSE and MAE values.
   
2. **Linear Assumptions**:
   - The regression model was limited in its ability to capture non-linear price movements, suggesting the need for more complex models (e.g., LSTM).

3. **Feature Engineering**:
   - Integrating moving averages improved model performance by adding critical trend-based information.

---

## Next Steps
1. **Implement Non-Linear Models**:
   - Explore time-series models like LSTM to capture Bitcoin’s non-linear and volatile nature.

2. **Expand Features**:
   - Incorporate additional features, such as external market indicators or sentiment analysis, to improve model accuracy.

3. **Refine Moving Averages**:
   - Experiment with different moving average windows (e.g., 100-day, 300-day) for enhanced feature relevance.

---

This project demonstrated the challenges of predicting cryptocurrency prices and the importance of feature engineering and advanced modeling techniques in tackling non-linear and volatile datasets.

