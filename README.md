# NYC Green Taxi Fare Prediction using ML Alogrithms in PySpark

## Project Overview

This project aims to predict fare amounts for NYC Green Taxi trips using machine learning techniques. I have employed various regression models to estimate trip fares based on features extracted from the dataset. The project includes data cleaning, feature engineering, model training, and evaluation using Linear Regression, Decision Tree Regressor, and Random Forest Regressor.

## Dataset

The dataset used in this project is from the NYC Taxi and Limousine Commission (TLC). It contains records of trips made by NYC Green Taxis. Here is a detailed explanation of the dataset columns:

- `vendorID`: Identifier for the taxi vendor.
- `lpepPickupDatetime`: Pickup datetime of the trip.
- `lpepDropoffDatetime`: Dropoff datetime of the trip.
- `tripDistance`: Distance of the trip in miles.
- `fareAmount`: Fare amount in USD.
- `extra`: Additional charges for the trip.
- `mtaTax`: MTA tax for the trip.
- `improvementSurcharge`: Improvement surcharge for the trip.
- `tipAmount`: Tip amount given in USD.
- `tollsAmount`: Amount of tolls paid for the trip.
- `totalAmount`: Total amount paid for the trip.
- `passengerCount`: Number of passengers in the trip.
- `pickupLocationId`: Pickup location identifier.
- `dropoffLocationId`: Dropoff location identifier.
- `tripType`: Type of the trip (e.g., street-hail or dispatch).
- `storeAndFwdFlag`: Flag indicating if the trip data was stored and forwarded.

## Objective

The objective of this project is to build and evaluate multiple regression models to predict the total fare amount for NYC Green Taxi trips. The models tested include:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

## Scope

- **Data Cleaning**: Handle missing values and drop unnecessary columns.
- **Feature Engineering**: Create new features from datetime columns and convert categorical features to numerical values.
- **Model Training**: Train and evaluate Linear Regression, Decision Tree Regressor, and Random Forest Regressor models.
- **Evaluation**: Assess model performance using metrics such as Root Mean Squared Error (RMSE) and R^2 score.

## Technologies Used

- **Apache Spark**: For data processing and machine learning tasks.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **PySpark MLlib**: For machine learning models and evaluation.

## Code Explanation

### 1. Initialize Spark Session

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("NYC_Taxi_Analysis").getOrCreate()
```
This initializes a Spark session to use Spark's DataFrame and machine learning capabilities.

### 2. Read the Dataset

```python
df = spark.read.csv('dbfs:/FileStore/shared_uploads/jameers2003@gmail.com/nyc_tlc_green.csv', header=True, inferSchema=True)
df.show(5)
```
Reads the dataset into a Spark DataFrame with automatic schema inference.

### 3. Show Data Schema

```python
df.printSchema()
df.dtypes
```
Displays the schema and data types of the DataFrame columns.

### 4. Count NULL Values

```python
from pyspark.sql.functions import col, count, when
null_counts = df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df.columns]
)
null_counts.show()
```
Counts the number of NULL values in each column.

### 5. Data Cleaning

```python
df = df.drop('pickupLongitude', 'pickupLatitude', 'dropoffLongitude', 'dropoffLatitude', 'ehailFee')
df.show()
```
Drops columns that are not needed for analysis.

### 6. Recheck NULL Values

```python
null_counts = df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df.columns]
)
null_counts.show()
```
Counts NULL values again after dropping columns.

### 7. Display Summary Statistics

```python
df.describe().show()
```
Shows summary statistics for numerical columns.

### 8. Convert to Pandas DataFrame

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pandas_df = df.toPandas()
```
Converts the Spark DataFrame to a Pandas DataFrame for more detailed analysis and visualization.

### 9. Feature Engineering

```python
pandas_df['lpepPickupDatetime'] = pd.to_datetime(pandas_df['lpepPickupDatetime'])
pandas_df['lpepDropoffDatetime'] = pd.to_datetime(pandas_df['lpepDropoffDatetime'])
pandas_df['trip_time_in_secs'] = (pandas_df['lpepDropoffDatetime'] - pandas_df['lpepPickupDatetime']).dt.total_seconds()
```
Converts datetime columns to Pandas datetime objects and calculates trip duration in seconds.

### 10. Visualize Data

```python
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tripDistance', y='trip_time_in_secs', data=pandas_df)
plt.title('Trip Distance vs. Trip Duration')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Duration (seconds)')
plt.show()
```
Creates scatter plots and heatmaps to visualize relationships and correlations in the data.

### 11. Feature Assembly

```python
from pyspark.ml.feature import VectorAssembler
feature_columns = [
    'tripDistance', 'passengerCount', 'tripType', 'improvementSurcharge', 'tollsAmount', 'tipAmount', 'extra', 'mtaTax'
]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df = assembler.transform(df)
```
Assembles feature columns into a single vector for use in machine learning models.

### 12. Prepare Data for Modeling

```python
df_lr = df.select("features", col("totalAmount"))
train_df, test_df = df_lr.randomSplit([0.8, 0.2])
```
Prepares data for training and testing by splitting it into feature vectors and labels.

## Model Training and Evaluation

### Linear Regression

```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='totalAmount')

# Train the model
lr_model = lr.fit(train_df)

# Make predictions
lr_predictions = lr_model.transform(test_df)

# Evaluate the model rmse value
evaluator_rmse = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='rmse')
lr_rmse = evaluator_rmse.evaluate(lr_predictions)
print(f"Root Mean Squared Error (RMSE) for Linear Regression: {lr_rmse}")

# Evaluate the model r2 score
evaluator_r2 = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='r2')
lr_r2 = evaluator_r2.evaluate(lr_predictions)
print(f"R2 Score for Linear Regression: {lr_r2}")

# Print the coefficients and intercept for linear regression
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
```

Trains and evaluates the Linear Regression model. It calculates RMSE and R^2 score to assess performance and prints the coefficients and intercept.

### Decision Tree Regressor

```python
from pyspark.ml.regression import DecisionTreeRegressor

# Initialize Decision Tree Regressor model
dt = DecisionTreeRegressor(featuresCol='features', labelCol='totalAmount')

# Train the model
dt_model = dt.fit(train_df)

# Make predictions on test data
dt_predictions = dt_model.transform(test_df)

# Evaluate the model rmse value
evaluator_rmse = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='rmse')
dt_rmse = evaluator_rmse.evaluate(dt_predictions)
print(f"Root Mean Squared Error (RMSE) for Decision Tree Regressor: {dt_rmse}")

# Evaluate the model r2 value
evaluator_r2 = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='r2')
dt_r2 = evaluator_r2.evaluate(dt_predictions)
print(f"R2 Score for Decision Tree Regressor: {dt_r2}")
```

Trains and evaluates the Decision Tree Regressor model. It calculates RMSE and R^2 score to assess performance.

### Random Forest Regressor

```python
from pyspark.ml.regression import RandomForestRegressor

# Initialize Random Forest Regressor model
rf = RandomForestRegressor(featuresCol='features', labelCol='totalAmount')

# Train the model
rf_model = rf.fit(train_df)

# Make predictions
rf_predictions = rf_model.transform(test_df)

# Evaluate the model rmse value
evaluator_rmse = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='rmse')
rf_rmse = evaluator_rmse.evaluate(rf_predictions)
print(f"Root Mean Squared Error (RMSE) for Random Forest Regressor: {rf_rmse}")

# Evaluate the model r2 score
evaluator_r2 = RegressionEvaluator(labelCol='totalAmount', predictionCol='prediction', metricName='r2')
rf_r2 = evaluator_r2.evaluate(rf_predictions)
print(f"R2 Score for Random Forest Regressor: {rf_r2}")
```

Trains and evaluates the Random Forest Regressor model. It calculates RMSE and R^2 score to assess performance and prints feature importances.

## Compare Model Performance

```python
models = {
    "Model": ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor'],
    "RMSE": [lr_rmse, dt_rmse, rf_rmse],
    "R2": [lr_r2, dt_r2, rf_r2]
}
model_scores = pd.DataFrame(models)
print(model_scores)
```

Compares the performance of different models based on RMSE and R^2 score.

## Choose the Best Model

```python
best_model = 'Linear Regression' if lr_rmse < dt_rmse and lr_rmse < rf_rmse else 'Decision Tree' if dt_rmse < lr_rmse and dt_rmse < rf_rmse else 'Random Forest'
print(f"The best model is: {best_model}")
```

Identifies the best-performing model based on the lowest RMSE.

## Conclusion

In this project, we have successfully built and evaluated three different regression models to predict NYC Green Taxi fares. The Linear Regression model, Decision Tree Regressor, and Random Forest Regressor were compared based on their RMSE and R^2 scores.

The results indicate that Linear Regression model performed the best, with the lowest RMSE and highest R^2 score. This model can be effectively used to estimate taxi fares, which can be useful for dynamic pricing and fare prediction applications in real-time scenarios.

By incorporating more features or using advanced techniques, the model's performance can potentially be improved further. Future work may involve integrating real-time data and experimenting with other algorithms to enhance accuracy.

## Applications

This model can be used by taxi companies or fare estimation services to predict trip fares accurately. It can help in:

- **Fare Estimation**: Providing passengers with accurate fare estimates.
- **Dynamic Pricing**: Adjusting prices based on trip characteristics and demand.
- **Service Improvement**: Analyzing fare trends to improve service quality.

## How It Can Be Extended

- **Integration with Real-Time Data**: Apply the model to real-time data for live fare predictions.
- **Feature Expansion**: Incorporate additional features such as weather conditions or traffic data.
- **Model Improvement**: Experiment with other machine learning algorithms or ensemble methods for better accuracy.

## Dataset Links

- **[NYC Taxi and Limousine Commission (TLC) Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)**

## Usage

To run this project, ensure you have a Spark environment set up. Follow these steps:

1. **Load the Dataset**: Place the dataset in the specified location.
2. **Run the Notebook**: Execute each cell in the Databricks notebook to process the data and train models.

## Contact

For any queries, feel free to reach out to me at [jameers2003@gmail.com](mailto:jameers2003@gmail.com).
