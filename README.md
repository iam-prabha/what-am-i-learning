# What Am I Learning? 🚀

A comprehensive repository of resources, code, and tools for Python, Data Science, Machine Learning, Deep Learning, and more. This repository serves as a personal learning journey and reference guide for various technologies and concepts.

## 📚 Table of Contents

- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Technologies Covered](#-technologies-covered)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Learning Path](#-learning-path)
- [Documentation](#-documentation)
- [Code Quality](#-code-quality)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🚀 Getting Started

This repository uses `uv` for fast and reliable Python package management.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone git@github.com:iam-prabha/what-am-i-learning.git
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
│   ├── 04_data_structures.ipynb
│   ├── 05_oops_concept.ipynb
│   ├── 06_numpy.ipynb              # NumPy for numerical computing
│   ├── 07_pandas.ipynb             # Pandas for data manipulation
│   ├── 08_matplotlib.ipynb         # Matplotlib for plotting
│   ├── 09_seaborn.ipynb            # Seaborn for statistical visualization
│   ├── 10_exception_handling.ipynb
│   ├── 11_multiprocessing.py
│   ├── 12_multithreading.py
│   ├── 13_decorator.py
│   ├── 14_generator.py
│   ├── python_reference.md         # Quick Python syntax reference
│   └── slides/                     # Learning slides and materials
├── ml/                             # Machine Learning
│   ├── supervised_learning/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── decision_tree/
│   │   ├── random_forest/
│   │   ├── svm/
│   │   ├── knn/
│   │   ├── naive_bayes/
│   │   ├── adaboost/
│   │   └── gradient_boosting/
│   └── unsupervised_learning/
│       ├── k_means_clustering/
│       └── pca/
├── deep_learning/                  # Deep Learning with TensorFlow
│   └── tensorflow/
│       ├── 00_tensorflow_fundamentals/
│       ├── 01_artificial_neural_networks/
│       ├── 02_first_neural_network/
│       └── 03_convolutional_neural_networks/
├── dsa/                           # Data Structures & Algorithms
│   ├── data_structures/
│   │   ├── 00_arrays_and_strings.py
│   │   ├── 01_linked_list.py
│   │   ├── 02_stacks.py
│   │   ├── 03_queue.py
│   │   ├── 04_hash_tables.py
│   │   └── 05_heap.py
│   └── algorithms/
│       ├── sorting/
│       ├── searching/
│       └── graph/
├── sql/                           # SQL queries and examples
│   ├── beginner/
│   │   ├── 00_parks_and_rec_create_db.sql
│   │   ├── 01_select_statement.sql
│   │   ├── 02_where_statement.sql
│   │   ├── 03_group_by_order_by.sql
│   │   ├── 04_having_vs_where.sql
│   │   └── 05_limit_and_aliasing.sql
│   ├── intermediate/
│   │   ├── 01_joins.sql
│   │   ├── 02_unions.sql
│   │   ├── 03_string_functions.sql
│   │   ├── 04_case_statements.sql
│   │   ├── 05_subqueries.sql
│   │   └── 06_window_functions.sql
│   └── advanced/
│       ├── 01_ctes.sql
│       ├── 02_temp_tables.sql
│       ├── 03_stored_procedures.sql
│       └── 04_triggers_and_events.sql
├── statistics/                    # Statistical concepts
├── web_scraping/                  # Web scraping examples
├── javascript/                    # JavaScript fundamentals
├── linux/                         # Linux administration and commands
│   └── arch-pacman.md            # Arch Linux Pacman cheat sheet
├── projects/                      # Complete data science projects
├── data/                          # Centralized sample datasets
├── docs/                          # Extended documentation
├── scripts/                       # Utility scripts and tools
└── pdf's/                         # Reference materials
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
- **Linux** - System administration and command-line tools (Arch Linux/Pacman)

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
# Add a new package
uv add package_name

# Add a development dependency
uv add --group dev package_name
```

### Updating Dependencies
```bash
# Update all dependencies
uv sync

# Update specific package
uv add package_name@latest
```

## 🔧 Troubleshooting

### Common Issues

#### Issue: `uv` command not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

#### Issue: Python version mismatch
```bash
# Check Python version
python --version

# Use specific Python version with uv
uv python install 3.12
```

#### Issue: Jupyter kernel not found
```bash
# Install kernel
uv run python -m ipykernel install --user --name=what-am-i-learning
```

#### Issue: Import errors in notebooks
```bash
# Ensure all dependencies are installed
uv sync

# Restart Jupyter kernel
# Kernel -> Restart Kernel in Jupyter
```

For more issues, check the [Issues](https://github.com/iam-prabha/what-am-i-learning/issues) page.

## 📦 Dependencies

This project uses `uv` for dependency management. Key packages include:

### Core Dependencies
- **Data Science**: numpy, pandas, matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow
- **Web Scraping**: requests, beautifulsoup4
- **Jupyter**: jupyter, ipykernel

### Development Dependencies
- **Testing**: pytest
- **Code Formatting**: black
- **Linting**: flake8
- **Type Checking**: mypy

To install development dependencies:
```bash
uv sync --group dev
```

See `pyproject.toml` for the complete list of dependencies and versions.

## 🎯 Learning Path

1. **Start with Python fundamentals** (`python/` directory)
2. **Explore data science tools** (NumPy, Pandas, Matplotlib, Seaborn)
3. **Dive into machine learning** (`ml/` directory)
4. **Learn deep learning** (`deep_learning/` directory)
5. **Practice algorithms** (`dsa/` directory)
6. **Work on projects** (`projects/`)

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for a detailed learning roadmap with time estimates.

## 📚 Documentation

This repository includes comprehensive documentation:

- **[README.md](README.md)** - Main overview and getting started guide (this file)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick navigation guide and learning path
- **[REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)** - Detailed list of structural improvements
- **[CODE_QUALITY_REPORT.md](CODE_QUALITY_REPORT.md)** - Code quality analysis and improvement recommendations
- **Directory READMEs** - Each major directory has its own README with specific guidance

## 🔍 Code Quality

This repository maintains good code quality standards. See [CODE_QUALITY_REPORT.md](CODE_QUALITY_REPORT.md) for:

- Comprehensive code analysis (**Grade: B+**)
- Best practices and recommendations
- Code improvement roadmap
- Testing guidelines
- Style guide compliance

### Quick Quality Check
```bash
# Install development tools
uv sync --group dev

# Format code
black .

# Lint code
flake8 .

# Type check
mypy .

# Run tests (when available)
pytest tests/
```

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/improvement`)
3. **Make your changes**
4. **Run quality checks**:
   ```bash
   black .
   flake8 .
   mypy .
   ```
5. **Commit your changes** (`git commit -m 'Add some improvement'`)
6. **Push to the branch** (`git push origin feature/improvement`)
7. **Open a Pull Request**

### Contribution Ideas
- Report issues or bugs
- Suggest new topics to cover
- Share additional resources
- Improve documentation
- Add unit tests
- Fix code quality issues

See [CODE_QUALITY_REPORT.md](CODE_QUALITY_REPORT.md) for areas that need improvement.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Various online courses and tutorials
- Open source communities
- Educational resources and documentation
- The Python and data science community
- Contributors and supporters

## 📞 Contact

- **GitHub**: [@iam-prabha](https://github.com/iam-prabha)
- **Repository**: [what-am-i-learning](https://github.com/iam-prabha/what-am-i-learning)

## 📊 Repository Stats

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

**Happy Learning!** 🎓✨

*This repository represents a continuous learning journey in the fields of data science, machine learning, and software development. Keep exploring, keep learning!*

---

### Quick Links
- 📖 [Quick Reference Guide](QUICK_REFERENCE.md) - Fast navigation and learning tips
- 🔧 [Code Quality Report](CODE_QUALITY_REPORT.md) - Quality analysis and improvements
- 📋 [Reorganization Summary](REORGANIZATION_SUMMARY.md) - Structural changes documentation