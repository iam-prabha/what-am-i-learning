# Converted from 07-pandas.ipynb

# ======================================================================
# # Core Data Structures: Series and DataFrame
# Pandas introduces two primary data structures:
# 
# 1. **Series:** A one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.). It's essentially a single column of data.
# 
#     Think of it like a single column in a spreadsheet, where each row has an index.
# 
# 2. **DataFrame:** A two-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or a SQL table, where each column is a Series. It's the most commonly used Pandas object.
# 
#     Think of it as the entire spreadsheet, with multiple columns and rows.
# 
# **Getting Started: Installation and Import**
# 
# First, you need to install Pandas if you haven't already:
# ======================================================================

# ======================================================================
# bash: `pip install pandas`
# ======================================================================

# ======================================================================
# Then, in your Python script or Jupyter Notebook, you typically import it like this:
# ======================================================================

# %%
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ======================================================================
# **Creating Data Structures**
# ======================================================================

# ======================================================================
# **Creating a Series**
# 
# You can create a Series from a list, a NumPy array, or a dictionary.
# ======================================================================

# %%
# From a list
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series from a list:")
print(s)

# From a Numpy array
arr = np.array([10, 20, 30, 40])
s_arr = pd.Series(arr)
print("\nSeries from a Numpy array:")
print(s_arr)

# From a dictionary (Keys become the index)
data = {"a": 100, "b": 200, "c": 300}
s_dict = pd.Series(data)
print("\nSeries froma dictionary")
print(s_dict)

# ======================================================================
# **Creating a DataFrame**
# 
# DataFrames can be created in several ways, most commonly from dictionaries of Series/lists or NumPy arrays.
# ======================================================================

# %%
# From a dictionary of lists/Series
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, 30, 35, 40],
    "City": ["New York", "Los Angeles", "Chicago", "Houston"],
}
df = pd.DataFrame(data)
print("\nDataFrame from a dictionary:")
print(df)

# Specifying index and columns
df_indexed = pd.DataFrame(
    data, index=["a", "b", "c", "d"], columns=["Age", "Name", "City"]
)
print("\nDataFrame with custom index and columns:")
print(df_indexed)

# From a list of dictionaries (each dictionary is a row)
data_list_of_dict = [
    {"Name": "Eve", "Age": 22, "City": "Miami"},
    {"Name": "Frank", "Age": 28, "City": "Boston"},
]
df_from_list_of_dict = pd.DataFrame(data_list_of_dict)
print("\nDataFrame from a list of dictionaries:")
print(df_from_list_of_dict)

# ======================================================================
# **Viewing Data**
# 
# Once you have a DataFrame, you'll want to inspect it.
# ======================================================================

# %%
# Create a sample DataFrame for demonstration
data = {
    "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "B": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "C": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
}
df = pd.DataFrame(data)

# Display the first 5 rows
print("df.head():")
print(df.head())

# Display the last 3 rows
print("\ndf.tail(3):")
print(df.tail(3))

# Get a concise summary of the DataFrame
print("\ndf.info():")
print(df.info())

# Get descriptive statistics
print("\ndf.describe():")
print(df.describe())

# Get the shape (number of rows, number of columns)
print("\ndf.shape:")
print(df.shape)

# Get the column names
print("\ndf.columns:")
print(df.columns)

# Get the index
print("\ndf.index:")
print(df.index)

# ======================================================================
# **Selection and Indexing**
# 
# Accessing specific data is crucial.
# ======================================================================

# ======================================================================
# **Selecting a Single Column**
# ======================================================================

# %%
print("Selecting columns 'A':")
print(df["A"])  # Return a Series

print("\nSelecting columns 'B' (alternative method):")
print(df.B)  # Works if column name is a valid Python identifier

# ======================================================================
# **Selecting Multiple Columns**
# ======================================================================

# %%
print("\nSelecting columns 'A' and 'C':")
print(df[["A", "C"]])  # Return a DataFrame

# ======================================================================
# **Selecting Rows by Label (`.loc[]`)**
# 
# `.loc[]` is primarily label-based.
# ======================================================================

# %%
# Select a single row by its index label (e.g., row with index 2)
print("\nSelecting row with index 2 using  .loc:")
print(df.loc[2])

# Select multiple rows by index labels
print("\nSlecting row with indicies 0, 2, 4 using .loc:")
print(df.loc[[0, 2, 4]])

# Select rows by index labels (inclusive)
print("\nSelecting rows from index 1 to 3 (inclusive) using .loc:")
print(df.loc[1:3])

# Select specific rows and columns by labels
print("\nSelecting rows 0, 1 and columns 'A' and 'C' using .loc:")
print(df.loc[[0, 1], ["A", "C"]])

# ======================================================================
# **Selecting Rows by Position (`.iloc[]`)**
# 
# `.iloc[]` is primarily integer-position based.
# ======================================================================

# %%
# Selec a single row by its integer position (e.g., the 3rd row, which has index 2)
print("\nSelecting row at position 2 using .iloc:")
print(df.iloc[2])

# Select multiple rows by integer positions
print("\nSelecting rows at positions 0, 2, 4 using .iloc:")
print(df.iloc[[0, 2, 4]])

# Select a slice of rows by integer positions (exclusive of the end)
print("\nSelecting rows from position 1 up to (but not including) 4 using .iloc:")
print(df.iloc[1:4])

# Select specific rows and columns by integer positions
print("\nSelecting rows at positions 0, 1 and columns at positions 0, 2 using .iloc:")
print(df.iloc[[0, 1], [0, 2]])

# ======================================================================
# **Boolean Indexing (Filtering)**
# 
# This is a very powerful way to select data based on conditions.
# ======================================================================

# %%
# Select row where column 'A' is greater than 5
print("\nRows where 'A' > 5:")
print(df[df["A"] > 5])

# Select rows where column 'C' is 'x'
print("\nRows where 'C' is 'x':")
print(df[df["C"] == "x"])

# combine multiple conditions
print("Row where 'A' > 5 and 'C' is 'x':")
print(df[df["A"] > 5 & (df["C"] == "x")])

print("\nRows where 'A' > 5 or 'B' > 18:")
print(df[(df["A"] < 3) | (df["B"] > 18)])

# Using .isin() for multiple values
print("\nRows where 'C' is 'x' or 'y':")
print(df[df["C"].isin(["x", "y"])])

# ======================================================================
# **Handling Missing Data**
# 
# Missing data is represented by `NaN` (Not a Number).
# ======================================================================

# %%
df_missing = df.copy()
df_missing.iloc[1, 0] = np.nan  # set A[1] to NaN
df_missing.iloc[4, 1] = np.nan  # set B[4] to NaN
df_missing.iloc[6, 2] = np.nan  # set C[6] to NaN

print("\nDataFrame with missing values:")
print(df_missing)

# Check for missing values
print("\nChecking for missing values:")
print(df_missing.isnull().sum())

# Drop rows with any missing values
print("\nDataFrame after dropping rows with any missing values (df_missing.dropna()):")
print(df_missing.dropna())

# Fill missing values with a specific value (e.g., 0)
print("\nDataFrame after filling missing values with 0 (df_missing.fillna(0)):")
print(df_missing.fillna(0))

# Fill missing values with the mean of the column
print("\nDataFrame after filling NaN in 'A' with its mean:")
print(df_missing["A"].fillna(df_missing["A"].mean()))

# ======================================================================
# **Operations**
# 
# Pandas allows for various operations.
# 
# **Basic Arithmetic Operations**
# ======================================================================

# %%
df_ops = df.copy()
df_ops["D"] = df_ops["A"] + df_ops["B"]  # Add two columns
print("\nDataFrame with new column 'D' (A + B):")
print(df_ops)

df_ops["E"] = df_ops["A"] * 2  # Multiply a column by a scalar
print("\nDataFrame with new column 'E' (A * 2):")
print(df_ops)

# ======================================================================
# **Aggregations**
# ======================================================================

# %%
# Mean of column 'A'
print("\nMean of column 'A':", df["A"].mean())

# Sum of column 'B'
print("Sum of column 'B':", df["B"].sum())

# Max value of column 'A'
print("Max of column 'A':", df["A"].max())

# Count of non-null values in column 'A'
print("Count of non-null in 'A':", df["A"].count())

# ======================================================================
# **Grouping Data (`.groupby()`)**
# 
# This is similar to SQL's GROUP BY.
# ======================================================================

# %%
# Group by 'C' and calculate the mean of 'A' and 'B' for each group
print("\nGroup by 'C' and calculate mean of 'A' and 'B':")
print(df.groupby("C").mean())

# Group by 'C' and count the occurrences of each unique value in 'C'
print("\nValue counts for column 'C':")
print(df["C"].value_counts())

# ======================================================================
# **Input/Output (I/O)**
# 
# Pandas can read and write data in various formats.
# ======================================================================

# %%
# To CSV
df.to_csv(
    "../data/my_data.csv", index=False
)  # index=False prevents writing the DataFrame index as a column
print("\nDataFrame saved to '../data/my_data.csv'")

# From CSV
df_from_csv = pd.read_csv("../data/my_data.csv")
print("\nDataFrame read from '../data/my_data.csv':")
print(df_from_csv.head())

# To Excel
# df.to_excel('my_data.xlsx', sheet_name='Sheet1', index=False)
# print("\nDataFrame saved to 'my_data.xlsx'")

# From Excel
# df_from_excel = pd.read_excel('my_data.xlsx', sheet_name='Sheet1')
# print("\nDataFrame read from 'my_data.xlsx':")
# print(df_from_excel.head())

# %%


