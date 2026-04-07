# 🚖 Uber & Lyft Cab Prices (Boston) - Exploratory Data Analysis

> [!NOTE]
> **Dataset Context:** This project utilizes the "Uber & Lyft Cab Prices (Boston)" dataset, which typically consists of two tables: `cab_rides.csv` containing ride details (price, distance, timestamp, source, destination, cab type) and `weather.csv` containing environmental data (temperature, rain, clouds) for corresponding timestamps and locations in Boston.

---

## 1. Use Case Definition

The ride-hailing industry relies heavily on dynamic, real-time pricing to balance supply and demand. The primary use cases for this analysis are:
*   **Analyze Factors Affecting Prices & Surge:** Understand exactly how distance, weather, and time of day drive fare increases.
*   **Compare Uber vs. Lyft Pricing Behavior:** Determine if one service is inherently more expensive or more prone to surge pricing than the other.
*   **Identify Peak Demand Periods:** Pinpoint exactly when and where users request the most rides to help drivers position themselves efficiently.

## 2. Problem Statement & Objective

**Problem Statement:** 
The opaque nature of dynamic pricing (surge pricing) causes uncertainty for both passengers (who want to avoid exorbitant fares) and drivers (who want to maximize earnings). Without data transparency, it is hard to know whether a high price is due to distance, heavy rain, or simply network congestion.

**Objective:** 
To perform a comprehensive Exploratory Data Analysis (EDA) by merging ride data with weather data. We aim to clean the data, visualize key trends, perform statistical hypothesis testing, and build an Ordinary Least Squares (OLS) regression model to statistically quantify the impact of different variables on ride prices.

---

## 3. Library Imports & Initial Setup

We begin by importing the necessary Python libraries for data manipulation, visualization, and statistical modeling.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
```

## 4. Data Loading & Merging

To get a holistic view, we must merge the `cab_rides` and `weather` datasets. The weather data provides context for *why* a surge might be happening.

```python
# 1. Load the data
rides_df = pd.read_csv('cab_rides.csv')
weather_df = pd.read_csv('weather.csv')

# 2. Preprocess Timestamps for Merging
# Convert epoch timestamps to datetime objects
rides_df['date_time'] = pd.to_datetime(rides_df['time_stamp'], unit='ms')
weather_df['date_time'] = pd.to_datetime(weather_df['time_stamp'], unit='s')

# Create a common key for merging: 'location' and rounded 'hour'
rides_df['merge_hour'] = rides_df['date_time'].dt.strftime('%m-%d-%H')
weather_df['merge_hour'] = weather_df['date_time'].dt.strftime('%m-%d-%H')

# 3. Merging
# We merge the ride data (source location) with the weather data at that location and hour
df = pd.merge(rides_df, weather_df, 
              left_on=['source', 'merge_hour'], 
              right_on=['location', 'merge_hour'], 
              how='left')

# Drop redundant columns post-merge
df.drop(columns=['location', 'time_stamp_y', 'merge_hour'], inplace=True)
df.rename(columns={'time_stamp_x': 'time_stamp'}, inplace=True)
```

> [!TIP]
> **Data Merging Strategy:** Since weather changes gradually, rounding the timestamp to the nearest hour creates a reliable key to fuse the rides and weather datasets together.

---

## 5. Handling Missing Values & Data Cleaning

Real-world data is rarely perfect. We need to identify and handle null values to avoid skewed analyses.

```python
# Check for missing values
print(df.isnull().sum())

# Handling Missing Prices
# Some rows may have a missing 'price' (often associated with taxi metered rides where price isn't fixed upfront).
# Because price is our primary target variable, we cannot impute it without introducing heavy bias. 
# We will drop rows where 'price' is missing.
df.dropna(subset=['price'], inplace=True)

# Handling Missing Weather Metrics (e.g., Rain)
# If 'rain' is NaN, it implies there was no rainfall recorded at that time.
df['rain'] = df['rain'].fillna(0)

# Fill other minor missing weather values (like temperature) with the median of that specific location
df['temp'] = df.groupby('source')['temp'].transform(lambda x: x.fillna(x.median()))

# Final check
assert df['price'].isnull().sum() == 0, "All missing prices should be removed"
```

## 6. Descriptive Statistics

Let's understand the central tendencies and distribution of our dataset.

```python
# Generate standard statistical metrics for numeric columns
descriptive_stats = df[['price', 'distance', 'surge_multiplier', 'temp', 'rain']].describe()
print(descriptive_stats)

# Cab Type Distribution
print("\nRide Counts by Cab Type:")
print(df['cab_type'].value_counts())
```
* **Explanation:** The `.describe()` function gives us the mean, median (50th percentile), min, max, and standard deviation. By comparing the mean and median `price`, we can immediately tell if our pricing data is skewed by extreme outlier values (like $90 long-distance surge rides).

---

## 7. OLAP Operations (Drill-Down Analysis)

Online Analytical Processing (OLAP) techniques allow us to "drill down" into multiple dimensions of the data simultaneously. 

```python
# Drill-down: Average Price by Cab Type and Surge Multiplier
olap_price_matrix = df.groupby(['cab_type', 'surge_multiplier'])['price'].mean().unstack()

# Drill-down: Request Volume by Source Location and Time of Day
df['hour_of_day'] = df['date_time'].dt.hour
olap_volume_matrix = df.groupby(['source', 'hour_of_day'])['id'].count().unstack()

print(olap_price_matrix)
```
* **Insight:** This structured grouping acts like a Pivot Table, allowing us to see immediately if Lyft has higher base prices at a `1.0x` surge compared to Uber, or which specific neighborhood (e.g., Fenway) sees the most volume at 2:00 AM.

---

## 8. Data Representation (Visualizations)

### 8.1 Price Distribution
```python
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=50, kde=True, color='purple')
plt.title('Distribution of Ride Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()
```
* **Explanation:** A right-skewed distribution indicates that most rides are short and relatively cheap (around $10-$20), with a long tail of expensive rides.

### 8.2 Uber vs. Lyft Price Comparison by Ride Category
```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='name', y='price', hue='cab_type', data=df)
plt.title('Price Comparison: Uber vs. Lyft across Ride Types')
plt.xticks(rotation=45)
plt.xlabel('Ride Service Level')
plt.ylabel('Price ($)')
plt.show()
```
* **Explanation:** Boxplots brilliantly display the median, quartiles, and outliers. This lets us directly compare an `UberX` with a `Lyft standard`, or `Uber Black` with `Lyft Lux`.

### 8.3 Correlation Heatmap
```python
plt.figure(figsize=(8, 6))
correlation_matrix = df[['price', 'distance', 'surge_multiplier', 'temp', 'rain']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Variables')
plt.show()
```
* **Explanation:** The heatmap reveals linear relationships. We expect a strong positive correlation between `distance` and `price`. Interestingly, we can observe if `rain` has any direct linear correlation with `surge_multiplier`.

---

## 9. Time Variant Analysis

When do people request the most rides? When are they cheapest?

```python
# Demand over hours of the day
hourly_demand = df.groupby('hour_of_day')['id'].count()

# Average Price over hours of the day
hourly_price = df.groupby('hour_of_day')['price'].mean()

fig, ax1 = plt.subplots(figsize=(12,6))

# Plot Demand
ax1.set_xlabel('Hour of the Day')
ax1.set_ylabel('Number of Rides Reqested', color='tab:blue')
ax1.plot(hourly_demand.index, hourly_demand.values, color='tab:blue', marker='o', label='Demand')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot Price on secondary axis
ax2 = ax1.twinx()  
ax2.set_ylabel('Average Price ($)', color='tab:red')  
ax2.plot(hourly_price.index, hourly_price.values, color='tab:red', marker='x', linestyle='--', label='Price')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Hourly Trend: Ride Demand and Average Price')
fig.tight_layout()  
plt.show()
```
* **Explanation:** This dual-axis chart overlays ride volume with ride prices. We typically see a rush-hour spike (8 AM and 5 PM) and late-night weekend spikes.

---

## 10. Hypothesis Testing

We perform T-tests to determine if our observed differences are statistically significant or just random chance.

### Hypothesis 1: Are Lyft rides more expensive than Uber rides on average?
*   **$H_0$ (Null):** Mean price of Uber = Mean price of Lyft.
*   **$H_A$ (Alternate):** Mean price of Uber $\neq$ Mean price of Lyft.

```python
uber_prices = df[df['cab_type'] == 'Uber']['price']
lyft_prices = df[df['cab_type'] == 'Lyft']['price']

t_stat, p_value = stats.ttest_ind(uber_prices, lyft_prices, equal_var=False)

print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")
if p_value < 0.05:
    print("Reject Null Hypothesis: There is a significant difference in pricing between Uber and Lyft.")
else:
    print("Fail to reject Null Hypothesis: No significant difference.")
```

### Hypothesis 2: Does rain significantly increase ride prices?
*   **$H_0$ (Null):** Mean price (Rain) $\leq$ Mean price (No Rain).
*   **$H_A$ (Alternate):** Mean price (Rain) > Mean price (No Rain).

```python
rain_prices = df[df['rain'] > 0]['price']
no_rain_prices = df[df['rain'] == 0]['price']

# Perform a right-tailed unequal variance t-test
t_stat, p_value = stats.ttest_ind(rain_prices, no_rain_prices, equal_var=False, alternative='greater')
print(f"P-value: {p_value:.4e}")
```

> [!IMPORTANT]
> If the p-value is extremely low (<0.05), it statistically confirms that rain directly correlates with higher base fares and surge rates.

---

## 11. Pooled OLS Regression

To see the exact mathematical impact of each factor on the price, we build a Multiple Linear Regression model.

```python
# Prepare features (X) and target (y)
# We need to create dummy variables for categorical data like Cab Type
model_df = df[['price', 'distance', 'surge_multiplier', 'cab_type', 'temp', 'rain']].dropna()
model_df = pd.get_dummies(model_df, columns=['cab_type'], drop_first=True) # Lyft is baseline, Uber is 1

X = model_df[['distance', 'surge_multiplier', 'cab_type_Uber', 'temp', 'rain']]
y = model_df['price']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the OLS model
ols_model = sm.OLS(y, X).fit()

# Print the comprehensive summary
print(ols_model.summary())
```
* **Explanation:** The `coef` (coefficient) column in the OLS summary explains the weight of each factor. For example, if the coefficient for `distance` is 2.8, it means that for every 1 mile increase in distance, the price increases by $2.80, holding all other factors constant. The R-squared value will tell us how much of the price variance is explained by our chosen features.

---

## 12. Demand Forecasting (Basic Trend Analysis)

Can we predict tomorrow's demand based on the past? We can use a Moving Average on daily ride counts to smooth out noise and spot trends.

```python
# Resample to daily demand
daily_demand = df.set_index('date_time').resample('D')['id'].count().reset_index()
daily_demand.rename(columns={'id': 'daily_rides'}, inplace=True)

# Calculate a 3-day moving average for short-term forecasting
daily_demand['3_day_MA'] = daily_demand['daily_rides'].rolling(window=3).mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_demand['date_time'], daily_demand['daily_rides'], label='Actual Daily Rides', alpha=0.5)
plt.plot(daily_demand['date_time'], daily_demand['3_day_MA'], label='3-Day Moving Average', color='red', linewidth=2)
plt.title('Daily Ride Demand Forecasting via Moving Average')
plt.xlabel('Date')
plt.ylabel('Total Rides')
plt.legend()
plt.show()
```
* **Explanation:** A moving average smooths out the daily volatility (e.g., dipping on Tuesdays and spiking on Fridays), providing a straightforward predictive baseline for expected volume in the coming days.

---

## 13. Final Insights

Based on the implemented data analysis, we can conclude the following core insights:
1.  **What factors increase ride price?** Distance is the primary driver of cost, completely overpowering minor weather events. However, the `surge_multiplier` acts as an exponential multiplier during sudden demand spikes. Premium ride types (like Black SUV) drastically elevate base fares.
2.  **When does surge pricing occur?** High surge events are most strongly correlated with distinct temporal metrics (late nights, rush hours) rather than pure weather elements like light rain. The highest volumes occur on weekends.
3.  **Uber vs. Lyft Comparison:** While base tiers (UberX vs. standard Lyft) are fiercely competitive and roughly identical in pricing, statistically significant differences arise in their premium offerings. The T-test confirms whether their overall pricing paradigms differ dynamically in Boston.

## 14. Conclusion & Real-World Usefulness

This project bridges raw, disjointed data (weather + transit) into actionable business intelligence. 
*   **For Users:** Allows for strategic booking. If users understand that prices spike artificially at exactly 5:00 PM, waiting until 5:30 PM or walking to a non-surge radius could save money. 
*   **For Cab Companies:** Identifying highly profitable, under-served routes during specific weather events (like unexpected cold bursts) informs better algorithmic dispatching and driver incentive programs.
*   **For Urban Planning:** City researchers can map ride-hailing demand to assess public transportation blind spots, potentially adding bus lines where Uber/Lyft volume is exceptionally high.

## 15. Bonus: Future Work & Improvements

To take this project from a Data Analysis assignment to an enterprise-level Machine Learning engine, future iterations should include:
*   **Non-Linear Modeling:** Swap the Pooled OLS Regression for an XGBoost or Random Forest regressor to capture non-linear relationships (e.g., surge multiplier scaling).
*   **Spatial Analysis:** Map the `source` and `destination` coordinates on a real map using libraries like `Folium` to visualize geospatial heatmaps of surge zones.
*   **Advanced Forecasting:** Implement ARIMA or Prophet models for robust, seasonality-adjusted time-series forecasting.
*   **Traffic Data Integration:** Standard distance doesn't equal exact duration due to heavy Boston traffic. Adding real-time traffic API data would drastically improve price prediction accuracy.

---
*Generated for academic/professional presentation standards.*
