# Converted from naive_bayes.ipynb

# ======================================================================
# ## Naive Bayes Theorm
# ======================================================================

# ======================================================================
# **concept:** Naive Bayes is a classification algorithm based on Bayes'Theorm with 'naive' assumption: it assumes that the presence of a particular features in a class is independent of the presence of any other feature. while this assumption is rarely true in rea;-world data, it often performs surprisingly well, especially for text classification.
# ======================================================================

# ======================================================================
# * **Goal:** classification.
# 
# * **How it works:** It calculates the probability of an event given prior knowledge, making the 'naive' assumption of feature independence. 
# ======================================================================

# ======================================================================
# **To Master Easier:** Imagine you're tying to figure out if an email is spam. if it contain words like 'viagra,', 'lottary,' and 'free,' Naive Bayes assumes these words contribute independently to the "spam" probability, even if they often appear together.
# ======================================================================

# ======================================================================
# **Real time uses cases:**
# 
# * **Spam Filtering:** Highly effective for classifying spam emails.
# 
# * **Sentiment Analysis:** Determining the sentiment of a piece of text (positive, negative, neutral).
# 
# * **Document Classification:** Categorizing news articles, reviews, etc.
# 
# * **Medical Diagnosis:** Predicting the likelihood of a disease given symptoms.
# ======================================================================

# ======================================================================
# #### Naive Bayes Theorm (code Example)
# ======================================================================

# ======================================================================
# **Dataset:** A synthetic dataset suitable for demonstrating Naive Bayes, like the make_classification dataset.
# ======================================================================

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Naive Bayes Theorm (GuassianNB) ---")

# %%
# 1. Create a synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"\nX sample: {X[:5]}, \n y sample: {y[:5]}")

# %%
# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"\n X_train_sample: {X_train[:5]}, \n y_train_sample: {y_train[:5]}")

# %%
# 2. Train the Gaussian Naive Bayes model
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# %%
# 3. Make predictions on the test set
y_pred_nb = model_nb.predict(X_test)
y_prob_nb = model_nb.predict_proba(X_test)[
    :, 1
]  # Probability estimates for the positive class
print(f"\n y_pred_nb: {y_pred_nb[:5]}")
print(f"\n y_prob_nb: {y_prob_nb[:5]}")

# %%
# 4. Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
class_report_nb = classification_report(y_test, y_pred_nb)

print(f"\nAccuracy: {accuracy_nb:.2f}")
print(f"\nConfusion Matrix:\n{conf_matrix_nb}")
print(f"\nClassification Report:\n{class_report_nb}")

# %%
print(
    "\nDiscussion: Naive Bayes is based on Bayes' theorem with a strong independence assumption. Gaussian Naive Bayes assumes features follow a Gaussian (normal) distribution. "
    "It's often very fast and performs surprisingly well, especially for text classification."
)

# %%
# visualise the actual data vs predicted data
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test[:, 0],
    y=X_test[:, 1],
    hue=y_test,
    style=y_pred_nb,
    palette="deep",
    alpha=0.7,
)
plt.title("Actual vs Predicted Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Classes")
plt.show()

# %%


