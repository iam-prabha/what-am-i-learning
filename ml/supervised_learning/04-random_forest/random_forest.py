# Converted from random_forest.ipynb

# ======================================================================
# ## Random Forest
# 
# **Theory/concept**
# ======================================================================

# ======================================================================
# Random Forest is an **ensemble learning method** for **classification** and that operates by constructing a multitude (a large number of people or things.) of **decision trees** at training time and outputting the class that is the made of the classes (classification) or mean prediction (regression) of the individual trees.
# ======================================================================

# ======================================================================
# The 'randomness' comes from two main sources:
# 
# 1. **Bagging (Bootstrap Aggregating):** Each tree in the forest is trained on a random subset of the training data (sampled with replacement). This means some data points might be picked multiple times, and some not at all, for a single tree's training set. This reduces variance.
# 
# 2. **Features Randomness:** when growing a each tree, at each split node, only a random subset of features is considered for the best split. This further decorrelates the trees, making the ensemble more robust.
# ======================================================================

# ======================================================================
# By combining many de-correlated trees, the ensemble significantly reduces overfitting and improves generalization compared to a single decision tree.
# ======================================================================

# ======================================================================
# **Use Cases**
# 
# * **Loan Default Prediction**
# 
# * **Medical Diagnosis**
# 
# * **Predicting Sales**
# 
# * **Anywhere where decision trees are used, but with higher robustness.**
# ======================================================================

# ======================================================================
# **Pros and Cons**
# 
# * **Pros:**
# 
# * **High accuracy:** Often performs very well in practice.
# 
# * **Reduces overfitting:** By averaging multiple trees and using randomness.
# 
# * **Handles large datasets with many features.**
# 
# * **Less prone to overfitting** than individual decision trees.
# 
# * **Can handle both numerical and categorical features** (though `scikit-learn` requires numerical input).
# 
# * **Provides feature importance scores**, indicating which features are most impactful.
# 
# **Cons:**
# 
# * **Less interpretable** than a single decision tree (black box).
# 
# * **Computationally more expensive** and slower than single decision trees, especially with a large number of trees.
# 
# * **Requires more memory** as it stores multiple trees.
# ======================================================================

# ======================================================================
# `scikit-learn` implementation
# ======================================================================

# %%
# RandomForestRegresssion for regression tasks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine  # multi-class classification dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# %%
# 1. Load dataset
wine = load_wine()
X = wine.data
y = wine.target

print(f"{X[:5]}\n")
print(f"{y[:5]}")

# %%
# 2. split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# No scaling needed for tree-based models typically

# 3. Instantiate and Train the Random Forest Model
# n_estimators: The number of trees in the forest. More trees generally means better performance, but slower.
# max_features: The number of features to consider when looking for the best split.
# max_depth: The maximum depth of the tree.
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
rf_model = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42
)
rf_model.fit(X_train, y_train)

# %%
# 4. Make Predictions
y_pred_rf = rf_model.predict(X_test)
print(f"Predictions: {y_pred[:5]}")

# %%
# 5. Evaluate the Model
print("\n--- Random Forest Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_rf, target_names=wine.target_names),
)

# %%
# Feature Importance (a nice benefit of Random Forests)
feature_importances = pd.Series(rf_model.feature_importances_, index=wine.feature_names)
print("\nTop 10 Feature Importances:\n", feature_importances.nlargest(10))

# Important Hyperparameters for RandomForestClassifier:
# - n_estimators: (int) The number of trees in the forest. (Typically 100-500)
# - criterion: ('gini', 'entropy') The function to measure the quality of a split.
# - max_depth: (int or None) The maximum depth of the tree. None means nodes are expanded until all leaves are pure.
# - min_samples_split: (int or float) The minimum number of samples required to split an internal node.
# - min_samples_leaf: (int or float) The minimum number of samples required to be at a leaf node.
# - max_features: (int, float, 'auto', 'sqrt', 'log2') The number of features to consider when looking for the best split.
#     'sqrt' (default for classification): max_features = sqrt(n_features)
#     'log2' (default for regression): max_features = log2(n_features)
# - bootstrap: (bool) Whether bootstrap samples are used when building trees. (Default: True)
# - n_jobs: (int) Number of jobs to run in parallel. -1 means use all processors.

# %%


