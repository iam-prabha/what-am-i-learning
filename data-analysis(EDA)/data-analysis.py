# %% Imports & display settings
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Helpful display settings
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

print("Python:", sys.version.splitlines()[0])
print("NumPy:", np.__version__, "pandas:", pd.__version__)

# %% Section 1 — NumPy fundamentals
# What is NumPy?
# - NumPy provides the ndarray: an N-dimensional array of homogeneous numeric types.
# - It is the foundation for most numerical computing in Python.

# Create arrays in common ways
a = np.array([1, 2, 3])
b = np.array([[1.0, 2.0], [3.0, 4.0]])
c = np.zeros((3, 4))  # zeros
d = np.ones((2, 3))  # ones
e = np.arange(0, 10, 2)  # like range
f = np.linspace(0, 1, 5)  # equally spaced points

print("\nShapes and dtypes:")
for name, arr in [("a", a), ("b", b), ("c", c), ("e", e), ("f", f)]:
    print(name, "shape:", arr.shape, "dtype:", arr.dtype, "values:", arr)

# %% Indexing, slicing, and views
arr = np.arange(12).reshape(3, 4)
print("\narr:\n", arr)

# Basic indexing
print("arr[0, 1] ->", arr[0, 1])

# Row slice is a view (modifying it can modify original)
row0 = arr[0]
row0[0] = 999
print("\nAfter modifying row0, arr:\n", arr)

# Use copy() when you need an independent array
arr = np.arange(12).reshape(3, 4)
slice_view = arr[1:3, 1:4]
slice_copy = arr[1:3, 1:4].copy()
slice_view[0, 0] = -1
print("\narr after modifying slice_view:\n", arr)
print("slice_copy remains unchanged:\n", slice_copy)

# %% Broadcasting and vectorized ops
x = np.array([1, 2, 3])
y = np.array([10])
print("\nx + y ->", x + y)  # broadcast y to shape of x

A = np.arange(6).reshape(2, 3)
B = np.array([1, 2, 3])
print("\nA:\n", A)
print("A + B (row-wise broadcast):\n", A + B)

# Elementwise arithmetic
print("np.sin(A):\n", np.sin(A))
print("A * 2:\n", A * 2)
print("A @ A.T (matrix multiplication):\n", A @ A.T)

# %% Random numbers & reproducibility
rng = np.random.default_rng(42)  # recommended random generator
samples = rng.normal(loc=0, scale=1, size=(5, 3))
print("\nRandom samples (5x3):\n", samples)

# %% Basic stats & linear algebra
data = rng.integers(0, 100, size=(6, 4))
print("\ndata:\n", data)
print("mean axis=0 (columns):", data.mean(axis=0))
print("mean axis=1 (rows):", data.mean(axis=1))
print("std (flattened):", data.std())

# Linear algebra example: eigenvalues
M = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eig(M)
print("\nEigenvalues:", vals)
print("Eigenvectors:\n", vecs)

# %% Exercises (try these interactively)
# 1) Create a 10x10 array of random integers in [0, 50) and compute the normalized version
#    where each column has mean 0 and std 1.
# 2) Using broadcasting, compute pairwise Euclidean distances between rows of a small dataset.

# %% pandas basics
# What is pandas?
# - pandas provides `Series` (1D labeled) and `DataFrame` (2D labeled) objects.
# - It's built on NumPy and is the go-to library for data analysis in Python.

# Create a Series and DataFrame
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="example_series")
df = pd.DataFrame(
    {
        "A": np.arange(5),
        "B": rng.choice(["x", "y", "z"], size=5),
        "C": rng.normal(size=5),
    }
)
print("\nSeries s:\n", s)
print("\nDataFrame df:\n", df)

# %% Constructing DataFrames from NumPy
arr = np.arange(12).reshape(4, 3)
df_from_arr = pd.DataFrame(arr, columns=["col1", "col2", "col3"])
print("\nDataFrame from NumPy array:\n", df_from_arr)

# %% Reading & writing (example using in-memory CSV)
# Create a CSV string (simulating reading from a file)
csv_string = df.to_csv(index=False)
print("\nCSV preview:\n", csv_string.splitlines()[:5])

df2 = pd.read_csv(StringIO(csv_string))
print("\nRead back df2:\n", df2)

# %% Inspecting data
print("\ndf2.info():")
print(df2.info())
print("\ndf2.describe():\n", df2.describe(include="all"))

# %% Selection — label and position based
print("\nSelect column 'A':\n", df["A"])
print("\nSelect rows 1:3 using iloc:\n", df.iloc[1:3])

# Boolean selection / filtering
mask = df["C"] > 0
print("\nRows where C > 0:\n", df[mask])

# %% Adding, modifying, dropping columns
df["D"] = df["A"] * 2
df["E"] = pd.Categorical(df["B"])
print("\nAfter adding D and E:\n", df)

df = df.drop(columns=["E"])  # drop a column
print("\nAfter dropping E:\n", df)

# %% Missing values handling
df_with_nan = df.copy()
df_with_nan.loc[2, "C"] = np.nan
print("\ndf_with_nan:\n", df_with_nan)
print("isnull:\n", df_with_nan.isnull())

# Fill or drop
print("filled C with mean:")
print(df_with_nan["C"].fillna(df_with_nan["C"].mean()))

# %% GroupBy — summarize by groups
df_long = pd.DataFrame(
    {
        "category": rng.choice(["apple", "banana", "cherry"], size=20),
        "price": rng.uniform(0.5, 5.0, size=20),
        "qty": rng.integers(1, 10, size=20),
    }
)
print("\ndf_long head:\n", df_long.head())

grouped = df_long.groupby("category").agg(
    {"price": ["mean", "min", "max"], "qty": "sum"}
)
print("\nGrouped summary:\n", grouped)

# %% Merging / joining
left = pd.DataFrame({"key": ["a", "b", "c"], "Lval": [1, 2, 3]})
right = pd.DataFrame({"key": ["a", "b", "d"], "Rval": [10, 20, 30]})
print("\nleft:\n", left)
print("right:\n", right)
merged = left.merge(right, on="key", how="outer")
print("\nmerged (outer):\n", merged)

# %% Datetime handling
dates = pd.date_range("2023-01-01", periods=6, freq="D")
ts = pd.Series(np.random.randn(6), index=dates)
print("\nTime series ts:\n", ts)
print("ts.resample('2D').mean():\n", ts.resample("2D").mean())

# %% Practical mini-project: create a realistic sample dataset and analyze it
rng = np.random.default_rng(0)
users = pd.DataFrame(
    {
        "user_id": np.arange(1, 101),
        "age": rng.integers(18, 70, size=100),
        "signup_date": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 365, size=100), unit="D"),
        "country": rng.choice(
            ["IN", "US", "UK", "DE", "BR"], size=100, p=[0.4, 0.2, 0.15, 0.15, 0.1]
        ),
    }
)
transactions = pd.DataFrame(
    {
        "txn_id": np.arange(1, 501),
        "user_id": rng.choice(users["user_id"], size=500),
        "amount": np.round(rng.exponential(scale=50.0, size=500), 2),
        "ts": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 365, size=500), unit="D"),
    }
)

print("\nusers.head():\n", users.head())
print("\ntransactions.head():\n", transactions.head())

# Merge and analyze
tx_user = transactions.merge(users, on="user_id", how="left")
print("\nMerged example:\n", tx_user.head())

# Top 5 users by total spend
totals = tx_user.groupby("user_id")["amount"].sum().sort_values(ascending=False).head(5)
print("\nTop 5 users by total spend:\n", totals)

# Spend by country
spend_by_country = (
    tx_user.groupby("country")["amount"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)
print("\nSpend by country:\n", spend_by_country)

# %% matplotlib is a powerful tool for data visualization
x = [1, 2, 3, 4, 5]
y = [1, 3, 5, 7, 9]

# plt.plot(x, y)
plt.title("basic title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.show()

plt.plot(x, y, color="red", marker="*", linewidth=1, markersize=6)
plt.show()
# %% matplotlib default settings
plt.figure(figsize=(10, 8))

x1 = [1, 2, 3, 4, 5]
x2 = [6, 7, 8, 9, 10]
x3 = [2, 4, 6, 8, 10]
x4 = [1, 3, 5, 7, 9]

plt.subplot(2, 2, 1)
plt.plot(x1, x2)
plt.title("first plot")

plt.subplot(2, 2, 2)
plt.plot(x2, x3)
plt.title("second plot")

plt.subplot(2, 2, 3)
plt.plot(x2, x3)
plt.title("third plot")

plt.subplot(2, 2, 4)
plt.plot(x3, x4)
plt.title("fourth plot")
plt.tight_layout()
plt.show()

# %% bar plot

categories = ["a", "b", "c", "d", "e"]

values = [10, 20, 30, 40, 50]

plt.bar(categories, values, color="purple")
# %% Histogram
data = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 2, 10, 10]
plt.hist(data, bins=5)
# %% Scatter plot
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
plt.scatter(a, b, color="green")
# %% pie chart
labels = ["a", "b", "c", "d"]
sizes = [10, 20, 30, 40]
colors = ["gold", "yellowgreen", "lightcoral", "lightskyblue"]
explode = (0.1, 0, 0, 0)

plt.pie(sizes, labels=labels, colors=colors, explode=explode)

# %% Seaborn Advanced Visualization
df = sns.load_dataset("tips")
df.head()
# %%
sns.scatterplot(x="total_bill", y="tip", data=df)
plt.title("scatter plot")
plt.show()
# %% line plot
sns.lineplot(x="size", y="total_bill", data=df)
plt.title("line plot")
plt.show()
# %% bar plot
sns.barplot(x="day", y="total_bill", data=df)
plt.title("bar plot")
plt.show()
# %% boxplot
sns.boxplot(x="day", y="total_bill", data=df)
plt.title("box plot")
plt.show()
# %% histgram
sns.histplot(df["total_bill"], bins=10, kde=True)
plt.title("histogram")
plt.show()
# %% pairplot takes numeric columns
sns.pairplot(df)

# %% kde plot
sns.kdeplot(df["total_bill"], fill=True)
# %% correlation matrix
corr = df[["total_bill", "tip", "size"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
