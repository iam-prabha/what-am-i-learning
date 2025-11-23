# Machine Learning

Comprehensive machine learning implementations and examples using scikit-learn.

## 📁 Structure

### Supervised Learning
Algorithms that learn from labeled data:

- **01-linear_regression/** - Predicting continuous values
  - `linear_regression_scikit-learn.ipynb`
- **02-logistic_regression/** - Binary and multiclass classification
  - `logistic_regression.ipynb`
- **03-decision_tree/** - Tree-based decision making
  - `decision_tree.ipynb`
- **04-random_forest/** - Ensemble of decision trees
  - `random_forest.ipynb`
- **05-naive_bayes/** - Probabilistic classification
  - `naive_bayes.ipynb`
- **06-svm/** - Support Vector Machines for classification
  - `svm.ipynb`
- **07-knn/** - K-Nearest Neighbors algorithm
  - `KNN.ipynb`
- **08-gradient_boosting/** - Gradient Boosting ensemble method
  - `gradient_boost.ipynb`
- **09-adaboost/** - Adaptive Boosting ensemble method
  - `adaptive_boosting.ipynb`

### Unsupervised Learning
Algorithms that find patterns in unlabeled data:

- **01-k_means_clustering/** - Clustering similar data points
  - `K-means-cluster.ipynb`
- **02-pca/** - Principal Component Analysis for dimensionality reduction
  - `PCA.ipynb`

## 🎯 Quick Start

Each algorithm directory contains:
- Implementation notebooks with scikit-learn
- Example datasets and use cases
- Visualization of results
- Performance metrics

### Running Notebooks
```bash
# From repository root
uv run jupyter notebook ml/

# Or navigate to specific algorithm
cd ml/supervised_learning/01-linear_regression
uv run jupyter notebook linear_regression_scikit-learn.ipynb
```

## 📚 Learning Path

### Recommended Order
1. **Linear Regression** - Start with the simplest supervised learning algorithm
2. **Logistic Regression** - Learn classification basics
3. **Decision Trees** - Understand tree-based models
4. **Random Forest** - Learn ensemble methods
5. **Naive Bayes** - Explore probabilistic models
6. **SVM** - Understand support vector machines
7. **KNN** - Learn instance-based learning
8. **Gradient Boosting & AdaBoost** - Master advanced ensemble techniques
9. **K-Means & PCA** - Explore unsupervised learning

## 🔗 Related Content

- Data preprocessing: `../statistics/`
- Deep learning: `../deep_learning/`
- Real-world projects: `../projects/`
- Python fundamentals: `../python/`
- Sample datasets: `../data/`
