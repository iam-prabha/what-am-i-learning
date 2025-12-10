# Converted from KNN.ipynb

# ======================================================================
# ## **K-Nearest Neighbors (KNN)**
# 
# **Theory/Concept**
# 
# K-Nearest Neighbors (KNN) is a simple, non-parametric, **lazy learning** algorithm used for both **classification** and **regression.**
# 
# * **Non-parametric:** It makes no assumptions about the underlying data distribution.
# 
# * **Lazy learning:** It does not explicitly build a model during the training phase. Instead, all computations are deferred until prediction time.
# ======================================================================

# ======================================================================
# **How it works (for classification):**
# 
# 1. When a new, unseen data point needs to be classified, KNN looks at its `K` nearest data points in the training set.
# 
# 2. It then assigns the new data point to the class that is most common among its `K` nearest neighbors (a majority vote).
# 
# 3. For **regression**, it would take the average of the `K` nearest neighbors' target values.
# 
# **Distance Metric:** The "nearest" part is determined by a distance metric, most commonly **Euclidean distance**. Other options include Manhattan distance or Minkowski distance.
# ======================================================================

# ======================================================================
# **Use Cases**
# 
# **Recommendation Systems:** Finding users with similar preferences.
# 
# **Image Recognition:** Simple image classification.
# 
# **Missing Value Imputation:** Estimating missing values based on neighbors.
# 
# **Pros and Cons**
# 
# **Pros:**
# 
# * **Simple to understand and implement.**
# 
# * **No training phase required** (lazy learner).
# 
# * **Can handle multi-class classification problems easily.**
# 
# * **Flexible to feature distributions.**
# 
# **Cons:**
# 
# * **Computationally expensive during prediction** for large datasets, as it needs to calculate distances to all training points.
# 
# * **Sensitive to the scale of features** (features with larger ranges will dominate the distance calculation).
# 
# * **Sensitive to irrelevant features.**
# 
# * **Sensitive to outliers.**
# 
# * **Performance degrades with high dimensionality** (curse of dimensionality).
# 
# * **Choosing an optimal K value can be challenging.**
# ======================================================================

# ======================================================================
# `scikit-learn` Implementation
# ======================================================================

# %%
from sklearn.neighbors import KNeighborsClassifier  # KNeighborsRegressor for regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer  # A binary classification dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
# 1. Load Data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# %%
# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# %%
# 3. Scale Features (CRUCIAL for KNN as it's distance-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# 4. Instantiate and Train the KNN Model
# n_neighbors: The number of neighbors to consider (K). This is the most important hyperparameter.
# weights: ('uniform', 'distance') How to weigh neighbors. 'uniform' gives equal weight, 'distance' weights closer neighbors more.
# metric: ('minkowski', 'euclidean', 'manhattan') The distance metric.
knn_model = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")
knn_model.fit(X_train_scaled, y_train)  # Training simply involves storing the data

# %%
# 5. Make Predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# %%
# 6. Evaluate the Model
print("\n--- K-Nearest Neighbors (KNN) Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_knn, target_names=cancer.target_names),
)

# Important Hyperparameters for KNeighborsClassifier:
# - n_neighbors: (int) Number of neighbors to use by default for kneighbors queries. (Most important: K)
# - weights: ('uniform' or 'distance')
#    'uniform': all points in each neighborhood are weighted equally.
#    'distance': points are weighted by the inverse of their distance.
# - algorithm: ('auto', 'ball_tree', 'kd_tree', 'brute') Algorithm used to compute the nearest neighbors.
# - metric: (str or callable) The distance metric to use. Default is 'minkowski' (whic

# %%


