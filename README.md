# What Am I Learning? 🚀

A comprehensive repository of resources, code, and tools for Python, Data Science, Machine Learning, Deep Learning, and more. This repository serves as a personal learning journey and reference guide for various technologies and concepts.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Technologies Covered](#technologies-covered)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Getting Started

This repository uses `uv` for fast and reliable Python package management.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd what-am-i-learning
```

2. Install dependencies:
```bash
uv sync
```

3. Start learning! 🎓

## 📁 Repository Structure

```
what-am-i-learning/
├── python/                          # Python fundamentals
│   ├── 01_data_types.ipynb         # Data types and variables
│   ├── 02_loops_and_conditional.ipynb
│   ├── 03_functions.ipynb
│   ├── 04_data_Structures.ipynb
│   ├── 05_oops_concept.ipynb
│   ├── 06_numpy.ipynb              # NumPy for numerical computing
│   ├── 07_Panda.ipynb              # Pandas for data manipulation
│   ├── 08_matplotlib.ipynb         # Matplotlib for plotting
│   ├── 09_seaborn.ipynb            # Seaborn for statistical visualization
│   ├── 10_exception_handling.ipynb
│   ├── 11_multiprocessing.py
│   ├── 12_multithreading.py
│   └── slides/                     # Learning slides and materials
├── ML/                             # Machine Learning
│   ├── Supervised_learning/
│   │   ├── Linear_regression/
│   │   ├── Logistic_regression/
│   │   ├── Decision_tree/
│   │   ├── Random_Forest/
│   │   ├── SVM/
│   │   ├── K-Nearest Neighbors (KNN)/
│   │   ├── Naive Bayes/
│   │   ├── AdaBoost/
│   │   └── Grading_Boosting/
│   └── unsupervised_learning/
│       ├── K-means Clustering/
│       └── PCA/
├── deep learning/                  # Deep Learning
│   ├── 00_fundamental of TensorFlow/
│   ├── 01_Artificial_neural_network/
│   ├── Build_our_first_neural_network/
│   └── Convolutional_neural_network/
├── DSA/                           # Data Structures & Algorithms
│   ├── 01_arrays_&_strings.ipynb
│   ├── 02_singly_&_doubly_linked.ipynb
│   ├── 03_hash_tables.ipynb
│   ├── 04_stacks_&_queues.ipynb
│   ├── 05_recursion.ipynb
│   └── reverse_str.py
├── SQL/                           # SQL queries and examples
│   ├── Beginner/
│   ├── Intermediate/
│   └── Advanced/
├── Statistics/                    # Statistical concepts
├── web scraping/                  # Web scraping examples
├── A_project_workflow/           # Complete data science project
└── PDF's/                        # Reference materials
```

## 🛠️ Technologies Covered

### Core Python
- **Data Types & Variables** - Understanding Python's type system
- **Control Structures** - Loops, conditionals, and flow control
- **Functions** - Function definition, parameters, and scope
- **Object-Oriented Programming** - Classes, inheritance, polymorphism
- **Data Structures** - Lists, dictionaries, tuples, sets
- **Exception Handling** - Error handling and debugging
- **Concurrency** - Multiprocessing and multithreading

### Data Science Stack
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Basic plotting and visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms

### Machine Learning
- **Supervised Learning**
  - Linear Regression
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - AdaBoost
  - Gradient Boosting
- **Unsupervised Learning**
  - K-Means Clustering
  - Principal Component Analysis (PCA)

### Deep Learning
- **TensorFlow Fundamentals** - Core concepts and operations
- **Artificial Neural Networks** - Building and training ANNs
- **Convolutional Neural Networks** - Image processing and computer vision
- **Keras** - High-level neural network API

### Additional Technologies
- **SQL** - Database queries and data manipulation
- **Statistics** - Statistical concepts and analysis
- **Web Scraping** - Data extraction from websites
- **Data Structures & Algorithms** - Problem-solving fundamentals

## 💻 Usage

### Running Jupyter Notebooks
```bash
uv run jupyter notebook
```

### Running Python Scripts
```bash
uv run python your_script.py
```

### Adding New Dependencies
```bash
uv add package_name
```

### Updating Dependencies
```bash
uv sync
```

## 📦 Dependencies

This project uses `uv` for dependency management. Key packages include:

- **Data Science**: numpy, pandas, matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow
- **Web Scraping**: requests, beautifulsoup4
- **Jupyter**: jupyter, ipykernel
- **Development**: pytest, black, flake8, mypy

See `pyproject.toml` for the complete list of dependencies.

## 🎯 Learning Path

1. **Start with Python fundamentals** (`python/` directory)
2. **Explore data science tools** (NumPy, Pandas, Matplotlib, Seaborn)
3. **Dive into machine learning** (`ML/` directory)
4. **Learn deep learning** (`deep learning/` directory)
5. **Practice algorithms** (`DSA/` directory)
6. **Work on projects** (`A_project_workflow/`)

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:
- Report issues or bugs
- Suggest new topics to cover
- Share additional resources
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Various online courses and tutorials
- Open source communities
- Educational resources and documentation
- The Python and data science community

---

**Happy Learning!** 🎓✨

*This repository represents a continuous learning journey in the fields of data science, machine learning, and software development. Keep exploring, keep learning!*