# Converted from Data_sicence_project_workflow.ipynb

# ======================================================================
# # import libraries
# ======================================================================

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# ======================================================================
# # read data from file 
# ======================================================================

# %%
data = pd.read_parquet("../data/yellow_tripdata_2025-03.parquet")

# %%
data

# %%
data.shape

# ======================================================================
# # Date exploration
# ======================================================================

# %%
data.columns

# %%
taxi_data = data[
    [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "total_amount",
    ]
]

# %%
taxi_data.head()

# %%
taxi_data.hist(figsize=(20, 10), bins=60)

# %%
taxi_data["RatecodeID"].value_counts()

# %%
taxi_data.reset_index().plot(
    kind="scatter", x="index", y="total_amount", figsize=(10, 5)
)

# %%
taxi_data[taxi_data["total_amount"] < 1000].reset_index().plot(
    kind="scatter", x="index", y="total_amount", figsize=(10, 5)
)

# ======================================================================
# Two things to deal with: negative values and very high values. Let's first look at negative values.
# ======================================================================

# %%
print(taxi_data[taxi_data["total_amount"] < 0].shape)
taxi_data[taxi_data["total_amount"] < 0].reset_index().plot(
    kind="scatter", y="total_amount", x="index", figsize=(10, 5)
)

# %%
taxi_data[taxi_data["total_amount"] < 0].head()

# %%
taxi_data[taxi_data["total_amount"] < 0]["payment_type"].value_counts()

# %%
taxi_data[taxi_data["total_amount"] < 0]["payment_type"].hist(bins=60, figsize=(10, 5))

# %%
print(taxi_data[taxi_data["total_amount"] == 0].shape)
taxi_data[taxi_data["total_amount"] == 0].head()

# %%
taxi_data[taxi_data["total_amount"] == 0]["trip_distance"].hist(
    bins=60, figsize=(10, 5)
)

# %%
taxi_data[taxi_data["total_amount"] < 0]["trip_distance"].value_counts()

# ======================================================================
# We can safely get rid of the negative values. What about the very high values?
# ======================================================================

# %%
taxi_data.reset_index().plot(
    kind="scatter", y="total_amount", x="index", figsize=(10, 5)
)

# %%
taxi_data[taxi_data["total_amount"] > 200].shape

# %%
taxi_data["total_amount"].mean()

# ======================================================================
# # Data Cleaning
# ======================================================================

# ======================================================================
# ## based on your data analysis figure out thing you wanna model to predict 
# ## Here, I have decide specific range within certain affordable price under $200 
# ======================================================================

# %%
taxi_data_filtered = taxi_data[
    (taxi_data["total_amount"] >= 0) & (taxi_data["total_amount"] <= 200)
]

# %%
print(taxi_data.shape)
taxi_data_filtered.shape

# ======================================================================
# checking missing values
# ======================================================================

# %%
taxi_data_filtered.isnull().sum()

# %%
total_rows = len(taxi_data_filtered)
missing_proportion = (
    taxi_data_filtered[["passenger_count", "RatecodeID"]].isna().sum() / total_rows
)
print(missing_proportion * 100)

# %%
# Impute with mode
taxi_data_filtered["passenger_count"].fillna(
    taxi_data_filtered["passenger_count"].mode()[0], inplace=True
)
taxi_data_filtered["RatecodeID"].fillna(
    taxi_data_filtered["RatecodeID"].mode()[0], inplace=True
)

# ======================================================================
# # Data preparation
# ======================================================================

# %%
taxi_data_prepared = taxi_data_filtered.copy()

# ======================================================================
# Making sure everything is in the right type
# ======================================================================

# %%
taxi_data_prepared.dtypes

# %%
taxi_data_prepared.loc[:, "RatecodeID"] = taxi_data_prepared["RatecodeID"].astype(str)
taxi_data_prepared.loc[:, "PULocationID"] = taxi_data_prepared["PULocationID"].astype(
    str
)
taxi_data_prepared.loc[:, "DOLocationID"] = taxi_data_prepared["DOLocationID"].astype(
    str
)
taxi_data_prepared.loc[:, "payment_type"] = taxi_data_prepared["payment_type"].astype(
    str
)

# %%
taxi_data_prepared.dtypes

# ======================================================================
# Transforming variables into the formats we need them
# ======================================================================

# %%
taxi_data_prepared.head()

# %%
taxi_data_prepared["transaction_date"] = pd.to_datetime(
    taxi_data_prepared["tpep_pickup_datetime"].dt.date
)
# -> we make it datetime again because it's very little use when it's just a string (can't compare, sort, etc.)
taxi_data_prepared["transaction_year"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.year
taxi_data_prepared["transaction_month"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.month
taxi_data_prepared["transaction_day"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.day
taxi_data_prepared["transaction_hour"] = taxi_data_prepared[
    "tpep_pickup_datetime"
].dt.hour

# %%
taxi_data_prepared

# %%
taxi_data_prepared.hist(bins=60, figsize=(20, 10))

# %%
taxi_data_prepared = taxi_data_prepared[taxi_data_prepared["transaction_year"] == 2025]
taxi_data_prepared = taxi_data_prepared[taxi_data_prepared["transaction_month"] == 3]

# ======================================================================
# Noting down categorical and numerical columns
# ======================================================================

# %%
categorial_columns = [
    "PULocationID",
    "transaction_date",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
]
numerical_columns = ["trip_distance", "total_amount"]
all_needed_columns = categorial_columns + numerical_columns

# %%
main_taxi_df = taxi_data_prepared[all_needed_columns]
print(main_taxi_df.shape)
main_taxi_df.head()

# ======================================================================
# Aggregate data points
# 
# Now is a good time to think about what we want to predict. Depending on this, we need to transform our data to have a certain format.
# ======================================================================

# %%
taxi_grouped_by_region = main_taxi_df.groupby(categorial_columns).mean().reset_index()
taxi_grouped_by_region["count_of_transaction"] = (
    main_taxi_df.groupby(categorial_columns).count().reset_index()["total_amount"]
)
print(taxi_grouped_by_region.shape)
taxi_grouped_by_region.head()

# %%
taxi_grouped_by_region["trip_distance"].hist(figsize=(10, 5), bins=100)

# %%
taxi_grouped_by_region["total_amount"].hist(figsize=(10, 5), bins=100)

# ======================================================================
# # Benchmark model
# ======================================================================

# %%
data_for_benchmark_model = taxi_grouped_by_region.copy()

# %%
data_for_benchmark_model

# %%
categorial_features_benchmark = [
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
]
input_features_benchmark = categorial_features_benchmark + ["trip_distance"]
target_features_benchmark = "total_amount"

# ======================================================================
# # Train-test split
# ======================================================================

# %%
from sklearn.model_selection import train_test_split

X_bench = data_for_benchmark_model[input_features_benchmark]
y_bench = data_for_benchmark_model[target_features_benchmark]

# one-hot encode
X_bench = pd.get_dummies(X_bench)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bench, y_bench, test_size=0.3, random_state=42
)

# ======================================================================
# # fit a model to the data
# ======================================================================

# %%
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=10)
tree.fit(X_train_b, y_train_b)

# ======================================================================
# # Model Evaluation
# ======================================================================

# %%
model_at_hand = tree

y_pred_b = model_at_hand.predict(X_test_b)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

print(f"mean_absolute_error : {mean_absolute_error(y_test_b, y_pred_b)}")
print(f"mean_squared_error : {mean_squared_error(y_test_b, y_pred_b)}")
print(f"root_mean_squared_error : {sqrt(mean_squared_error(y_test_b, y_pred_b))}")
print(f"r2 : {r2_score(y_test_b, y_pred_b)}")

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(kind="scatter", figsize=(20, 10), x="true", y="pred")

# ======================================================================
# Could this be too good to be true?
# ======================================================================

# ======================================================================
# # Fix problems
# ======================================================================

# %%
# Here trip_distance i not include because that optional
categorial_features_benchmark = [
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
]
input_features_benchmark = categorial_features_benchmark
target_features_benchmark = "total_amount"

# Train and test split
from sklearn.model_selection import train_test_split

X_bench = data_for_benchmark_model[input_features_benchmark]
y_bench = data_for_benchmark_model[target_features_benchmark]

# one-hot encode
X_bench = pd.get_dummies(X_bench)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bench, y_bench, test_size=0.3, random_state=42
)

# Fit a model
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=10)
tree.fit(X_train_b, y_train_b)

# Evaluate model
model_at_hand = tree

y_pred_b = model_at_hand.predict(X_test_b)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

print("mean_absolute_error", mean_absolute_error(y_test_b, y_pred_b))
print("mean_squared_error", mean_squared_error(y_test_b, y_pred_b))
print("root_mean_squared_error", sqrt(mean_squared_error(y_test_b, y_pred_b)))
print("r2", r2_score(y_test_b, y_pred_b))

# %%
data = {"true": y_test_b, "pred": y_pred_b}
results = pd.DataFrame(data)

results.plot(kind="scatter", figsize=(20, 10), x="true", y="pred")

# ======================================================================
# # Featuring engineering
# ======================================================================

# %%
taxi_grouped_by_region.head()

# %%
data_with_new_features = taxi_grouped_by_region.copy()

# ======================================================================
# **Date-related features**
# ======================================================================

# %%
data_with_new_features["transaction_week_day"] = data_with_new_features[
    "transaction_date"
].dt.weekday
data_with_new_features["weekend"] = data_with_new_features[
    "transaction_week_day"
].apply(lambda x: True if x == 5 or x == 6 else False)

# %%
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holiday = cal.holidays(start="2025", end="2025").date

data_with_new_features["is_holiday"] = data_with_new_features["transaction_date"].isin(
    holiday
)

# %%
data_with_new_features.head()

# %%


