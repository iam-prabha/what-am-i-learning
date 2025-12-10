# Converted from gradient_boost.ipynb

# ======================================================================
# ### **Gradient Boosting**
# ======================================================================

# ======================================================================
# **concept:** Gradient Boosting like AdaBoost, is a sequential ensemble method.However, instead of focusing on misclassified samples, it focuses on the "residuals" (the errors) from the previous model. Each new tree tries to correct the errors of the previous ensemble. It uses a gradient descent optimization algorithm to minimize the loss function.
# 
# * **Goal:** Classification and Regression.
# 
# * **How it works:** It builds an ensemble of weak learners (typically decision trees) sequentially, where each new learner corrects the errors made by the previous ones.
# 
# **To Master Easier:** Think of a team trying to hit a target. The first team member takes a shot and misses. The next team member then tries to correct the error of the first shot, aiming closer to the target. This continues, with each team member trying to reduce the remaining error.
# ======================================================================

# ======================================================================
# **Real-time Use Cases:**
# 
# * **Web Search Ranking:** Optimizing search results by learning from user clicks.
# 
# * **Financial Modeling:** Predicting stock movements, credit risk.
# 
# * **Fraud Detection:** Highly effective due to its ability to capture complex patterns.
# 
# * **Customer Lifetime Value (CLV) Prediction:** Predicting the future value of a customer.
# ======================================================================

# ======================================================================
# ### Implementation 
# 
# **Dataset:** A synthetic regression dataset
# ======================================================================

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# %%
print("\n--- Gradient Boosting (Regressor) ---")

# %%
# 1. create a synthetic dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=5, random_state=42)
print("Dataset shape:", X.shape, y.shape)
print("First 5 samples of X:\n", X[:5])
print("First 5 samples of y:", y[:5])

# %%
# 2. split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining set shape:", X_train.shape, y_train.shape)
print("\nTesting set shape:", X_test.shape, y_test.shape)
print("\nFirst 5 samples of X_train:\n", X_train[:5])
print("First 5 samples of y_train:", y_train[:5])

# %%
# 3. Train the Model
model_gbr = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model_gbr.fit(X_train, y_train)
print("\nModel trained successfully.")

# %%
# 4. Make predictions
y_pred_gbr = model_gbr.predict(X_test)
print("\nFirst 5 predictions:", y_pred_gbr[:5])

# %%
# 5. Evaluate the Model
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

# %%
print(f"Root Mean Squared Error (RMSE): {rmse_gbr:.2f}")
print(f"R-squared: {r2_gbr:.2f}")

print(
    "\nDiscussion: Gradient Boosting builds an ensemble of weak learners (typically decision trees) sequentially. Each new tree attempts to correct the errors (residuals) of the previous ensemble. It's highly powerful and widely used for both regression and classification."
)

# %%


