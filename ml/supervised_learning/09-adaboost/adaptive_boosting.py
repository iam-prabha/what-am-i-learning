# Converted from adaptive_boosting.ipynb

# ======================================================================
# ### **Adaptive Boosting**
# ======================================================================

# ======================================================================
# **concept:** AdaBoost is another ensemble technique, but instead building trees independently like Random forest, it builds them sequentially. It focuses on the samples that were misclassified by the previous 'weak' learners (often simple decision stumps - very short trees). It gives more weight to thes misclassified samples in the next iteration, forcing the subsequent learners to pay more attention to them.
# ======================================================================

# ======================================================================
# * **Goal:** Classification (primarily) and Regression.
# 
# * **How it works:** It combines multiple "weak" learners to create a strong learner. It iteratively re-weights misclassified instances.
# ======================================================================

# ======================================================================
# **To Master Easier:** Imagine you're a teacher and have a class struggling with a particular concept. AdaBoost is like focusing extra attention and different teaching methods on the students who are still struggling, ensuring they eventually grasp the concept.
# ======================================================================

# ======================================================================
# **Real-time Use Cases:**
# 
# * **Face Detection:** Pioneering algorithm in real-time face detection (e.g., in cameras).
# 
# * **Spam Filtering:** Improving the accuracy of spam detection.
# 
# * **Medical Diagnosis:** Enhancing diagnostic accuracy by focusing on difficult cases.
# 
# * **Customer Relationship Management (CRM):** Predicting customer behavior with higher precision.
# ======================================================================

# ======================================================================
# **AdaBoost Implementation**
# 
# **Dataset:** A synthetic classification dataset.
# ======================================================================

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # AdaBoost often uses shallow trees
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- AdaBoost Classifier ---")

# %%
# 1. create a asynthetic classification dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=0
)

# %%
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# %%
# 2. Train the model
model_ab = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42,
    learning_rate=0.01,
)
model_ab.fit(X_train, y_train)
print("Model trained successfully!")
print(f"Number of estimators: {model_ab.n_estimators}")

# %%
# 3. make predictions on the test set
y_pred = model_ab.predict(X_test)
print(f"Predictions: {y_pred[:10]}")  # Display first 10 predictions

# %%
# 4. Evaluate the model
accuracy_ab = accuracy_score(y_test, y_pred)
conf_matrix_ab = confusion_matrix(y_test, y_pred)
class_report_ab = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_ab:.2f}")
print(f"Confusion Matrix: {conf_matrix_ab}")
print(f"Classification Report: {class_report_ab}")

# %%
print(
    "\nDiscussion: AdaBoost sequentially builds 'weak' learners (like shallow decision trees). It focuses on misclassified samples by giving them more weight in subsequent iterations, leading to a strong combined classifier. It's good for improving the performance of simple models."
)

# %%


