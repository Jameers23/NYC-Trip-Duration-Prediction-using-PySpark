# NYC Taxi Trip Duration Prediction Using Machine Learning Models (PySpark)

## Objective
The primary objective of this project is to predict the trip duration of New York City taxi rides using various machine learning regression models. The project aims to compare the performance of Linear Regression, Decision Tree, and Random Forest models and determine the most accurate model based on Root Mean Squared Error (RMSE).

## Scope of the Project
1. **Data Preprocessing:** Cleaning and transforming the NYC taxi trip dataset to make it suitable for machine learning.
2. **Feature Engineering:** Creating new features from the existing data to enhance model performance.
3. **Model Training and Evaluation:** Training multiple machine learning models and evaluating their performance.
4. **Visualization:** Visualizing the relationships in the data and the performance of the models.
5. **Comparison and Selection:** Comparing the models based on their RMSE and selecting the best model.

## Dataset
The dataset used in this project is the NYC TLC Green Taxi Trip Records dataset. It contains trip records for green taxis in New York City, including information such as pickup and dropoff times and locations, trip distances, fare amounts, and passenger counts.

### Dataset Details
- **vendorID:** A code indicating the provider associated with the trip record.
- **lpepPickupDatetime:** The date and time when the meter was engaged.
- **lpepDropoffDatetime:** The date and time when the meter was disengaged.
- **storeAndFwdFlag:** This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server. (Y = store and forward; N = not a store and forward trip)
- **rateCodeID:** The final rate code in effect at the end of the trip.
- **pickupLongitude:** The longitude where the meter was engaged.
- **pickupLatitude:** The latitude where the meter was engaged.
- **dropoffLongitude:** The longitude where the meter was disengaged.
- **dropoffLatitude:** The latitude where the meter was disengaged.
- **passengerCount:** The number of passengers in the vehicle. This is a driver-entered value.
- **tripDistance:** The elapsed trip distance in miles reported by the taximeter.
- **fareAmount:** The time-and-distance fare calculated by the meter.
- **extra:** Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges.
- **mtaTax:** $0.50 MTA tax that is automatically triggered based on the metered rate in use.
- **tipAmount:** Tip amount – This field is automatically populated for credit card tips. Cash tips are not included.
- **tollsAmount:** Total amount of all tolls paid in trip.
- **ehailFee:** The total amount of all fees to e-hail the trip.
- **improvementSurcharge:** $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015.
- **totalAmount:** The total amount charged to passengers. Does not include cash tips.

- **Dataset Link:** [NYC TLC Green Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## Technologies Used
- **Programming Language:** Python
- **Data Processing:** PySpark
- **Machine Learning:** PySpark MLlib
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas

## Applications
- **Traffic Management:** Predicting trip durations can help in optimizing routes and managing traffic congestion.
- **Fare Estimation:** Enhancing fare estimation models for better pricing strategies.
- **Service Optimization:** Improving the efficiency of taxi services by predicting trip times accurately.
- **Urban Planning:** Assisting urban planners in understanding transportation patterns and making informed decisions.

## Future Extensions
1. **Incorporate Additional Data:** Include weather data, traffic data, and special events to improve model accuracy.
2. **Use Advanced Models:** Experiment with more advanced machine learning models such as Gradient Boosting Machines or Neural Networks.
3. **Real-time Predictions:** Implement real-time trip duration prediction for live data.
4. **Deploy the Model:** Create a web or mobile application for users to predict trip durations on-the-go.

## Code Explanation

### Data Loading and Preprocessing

```python
from pyspark.sql import SparkSession

#Initialize Spark Session
spark = SparkSession.builder.appName("NYC_Taxi_Analysis").getOrCreate()

# Load the dataset
df = spark.read.csv('nyc_tlc_green.csv', header=True, inferSchema=True)

# Show the schema of the DataFrame
df.printSchema()

df.dtypes

# Display the first few rows
df.show(5)

from pyspark.sql.functions import col, count, when

# Count NULL values in each column
null_counts = df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df.columns]
)
null_counts.show()

df = df.drop('pickupLongitude', 'pickupLatitude', 'dropoffLongitude', 'dropoffLatitude', 'ehailFee')
df.show()

# Count NULL values again in each column
null_counts = df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df.columns]
)
null_counts.show()

df.dtypes
```

### Feature Engineering

```python
from pyspark.ml.feature import StringIndexer

# Convert categorical columns to numerical
# Initialize the StringIndexer for the 'storeAndFwdFlag' column
indexer = StringIndexer(inputCol="storeAndFwdFlag", outputCol="storeAndFwdFlagIndex")

# Fit the indexer to the DataFrame and transform it
df = indexer.fit(df).transform(df)

# Show the transformed DataFrame with new columns
df.show(5)

# Drop original categorical columns
df = df.drop('storeAndFwdFlag')
df.show()

df.dtypes

from pyspark.sql.functions import hour, month, dayofweek

# Create new columns: hour, month, day of week
df = df.withColumn('hour', hour(df['lpepPickupDatetime']))
df = df.withColumn('month', month(df['lpepPickupDatetime']))
df = df.withColumn('dayofweek', dayofweek(df['lpepPickupDatetime']))

df.show(5)

# Display summary statistics
df.describe().show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to pandas DataFrame
pandas_df = df.toPandas()

# Convert 'lpepPickupDatetime' and 'lpepDropoffDatetime' to pandas datetime if not already
pandas_df['lpepPickupDatetime'] = pd.to_datetime(pandas_df['lpepPickupDatetime'])
pandas_df['lpepDropoffDatetime'] = pd.to_datetime(pandas_df['lpepDropoffDatetime'])

# Calculate trip duration in seconds
pandas_df['trip_time_in_secs'] = (pandas_df['lpepDropoffDatetime'] - pandas_df['lpepPickupDatetime']).dt.total_seconds()

pandas_df.head()

# Scatter plot of trip distance vs. trip duration
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tripDistance', y='trip_time_in_secs', data=pandas_df)
plt.title('Trip Distance vs. Trip Duration')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Duration (seconds)')
plt.show()

# Scatter plot of fare amount vs. trip duration
plt.figure(figsize=(12, 6))
sns.scatterplot(x='fareAmount', y='trip_time_in_secs', data=pandas_df, alpha=0.5)
plt.title('Fare Amount vs. Trip Duration')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Trip Duration (seconds)')
plt.show()

# Compute the correlation matrix
corr = pandas_df[['tripDistance', 'trip_time_in_secs', 'fareAmount', 'extra', 'mtaTax', 'improvementSurcharge', 'tipAmount', 'tollsAmount', 'totalAmount']].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Convert pandas DataFrame to PySpark DataFrame
pandas_df_spark = spark.createDataFrame(pandas_df)

pandas_df_spark.printSchema()

pandas_df_spark.show()

# Select the required columns from pandas_df_spark for the join
pandas_df_spark = pandas_df_spark.select('vendorID', 'lpepPickupDatetime', 'lpepDropoffDatetime', 'trip_time_in_secs')

# Perform the join operation
df = df.join(pandas_df_spark, on=['vendorID', 'lpepPickupDatetime', 'lpepDropoffDatetime'], how='inner')

df.show(5)

# Rename 'trip_time_in_secs' to 'label'
df = df.withColumnRenamed('trip_time_in_secs', 'label')

# Select features and label
feature_columns = [
    'tripDistance', 'hour', 'month', 'dayofweek', 'puLocationId', 'doLocationId',
    'passengerCount', 'fareAmount', 'extra', 'mtaTax', 'tipAmount', 'tollsAmount',
    'improvementSurcharge', 'totalAmount', 'storeAndFwdFlagIndex'
]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df = assembler.transform(df)

df.show()

# Select the label and features
df_lr = df.select("features", col("label"))
df_lr.show()

# Split the data into training

 and test sets
train_df, test_df = df_lr.randomSplit([0.8, 0.2])

from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='label')

# Fit the model to the training data
lr_model = lr.fit(train_df)

# Make predictions on the test data
lr_predictions = lr_model.transform(test_df)

# Evaluate the model using RMSE
lr_evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
lr_rmse = lr_evaluator.evaluate(lr_predictions)
print(f"Root Mean Squared Error (RMSE) for Linear Regression: {lr_rmse}")

# Convert Spark DataFrame to Pandas DataFrame for plotting
lr_predictions_pd = lr_predictions.select('label', 'prediction').toPandas()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(lr_predictions_pd['label'], lr_predictions_pd['prediction'], alpha=0.5)
plt.plot([lr_predictions_pd['label'].min(), lr_predictions_pd['label'].max()],
         [lr_predictions_pd['label'].min(), lr_predictions_pd['label'].max()],
         color='red', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()

# Initialize the Decision Tree Regressor
dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')

# Fit the model to the training data
dt_model = dt.fit(train_df)

# Make predictions on the test data
dt_predictions = dt_model.transform(test_df)

# Evaluate the model using RMSE
dt_evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
dt_rmse = dt_evaluator.evaluate(dt_predictions)
print(f"Root Mean Squared Error (RMSE) for Decision Tree: {dt_rmse}")

# Convert Spark DataFrame to Pandas DataFrame for plotting
dt_predictions_pd = dt_predictions.select('label', 'prediction').toPandas()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(dt_predictions_pd['label'], dt_predictions_pd['prediction'], alpha=0.5)
plt.plot([dt_predictions_pd['label'].min(), dt_predictions_pd['label'].max()],
         [dt_predictions_pd['label'].min(), dt_predictions_pd['label'].max()],
         color='red', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree: Actual vs Predicted')
plt.show()

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(featuresCol='features', labelCol='label')

# Fit the model to the training data
rf_model = rf.fit(train_df)

# Make predictions on the test data
rf_predictions = rf_model.transform(test_df)

# Evaluate the model using RMSE
rf_evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
rf_rmse = rf_evaluator.evaluate(rf_predictions)
print(f"Root Mean Squared Error (RMSE) for Random Forest: {rf_rmse}")

# Convert Spark DataFrame to Pandas DataFrame for plotting
rf_predictions_pd = rf_predictions.select('label', 'prediction').toPandas()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(rf_predictions_pd['label'], rf_predictions_pd['prediction'], alpha=0.5)
plt.plot([rf_predictions_pd['label'].min(), rf_predictions_pd['label'].max()],
         [rf_predictions_pd['label'].min(), rf_predictions_pd['label'].max()],
         color='red', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted')
plt.show()

# Compare the performance of the models
rmse_scores = {
    "Linear Regression": lr_rmse,
    "Decision Tree": dt_rmse,
    "Random Forest": rf_rmse
}

print(f"RMSE Scores: {rmse_scores}")

# Choose the best model based on RMSE
best_model = min(rmse_scores, key=rmse_scores.get)
print(f"The best model is: {best_model}")
```

## Conclusion
This project demonstrates the process of predicting NYC taxi trip durations using various machine learning models, including Linear Regression, Decision Tree, and Random Forest. The project involves data preprocessing, feature engineering, model training, and evaluation. Based on RMSE, the best-performing model is selected to provide accurate trip duration predictions.
