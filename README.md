# What Am I Learning? 🚀

A comprehensive repository of resources, code, and tools for Python, Data Science, Machine Learning, Deep Learning, and more. This repository serves as a personal learning journey and reference guide for various technologies and concepts.

## 📚 Table of Contents

- [Overview](#-overview)
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

## 🎯 Overview

This repository is a comprehensive learning resource covering:

- **Python Fundamentals** - From basics to advanced concepts
- **Data Science** - NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning** - Supervised and unsupervised learning algorithms
- **Deep Learning** - Neural networks, CNNs, RNNs using PyTorch
- **Data Structures & Algorithms** - Implementation and practice
- **SQL** - Database queries from beginner to advanced
- **Statistics** - Statistical concepts and analysis
- **Web Scraping** - Data extraction techniques
- **Projects** - End-to-end data science workflows

Each directory contains well-organized learning materials, code examples, and documentation to support your learning journey.

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

### Quick Start Guide

- **New to Python?** Start with `python/01-data_types.py`
- **Want to learn ML?** Check out `ml/supervised_learning/`
- **Interested in Deep Learning?** Begin with `deep_learning/01-neural-network/`
- **Need SQL practice?** Start with `sql/beginner/`
- **Working on a project?** See `projects/Data_sicence_project_workflow.py`

## 📁 Repository Structure

```
what-am-i-learning/
├── python/                          # Python fundamentals
│   ├── 01-data_types.py            # Data types and variables
│   ├── 02-loops_and_conditional.py
│   ├── 03-functions.py
│   ├── 04-data_structures.py
│   ├── 05-oops_concept.py
│   ├── 06-numpy.py                 # NumPy for numerical computing
│   ├── 07-pandas.py                # Pandas for data manipulation
│   ├── 08-matplotlib.py            # Matplotlib for plotting
│   ├── 09-seaborn.py               # Seaborn for statistical visualization
│   ├── 10-exception_handling.py
│   ├── 11-multiprocessing.py
│   ├── 12-multithreading.py
│   ├── 13-decorator.py
│   ├── 14-generator.py
│   ├── pattern.py                  # Pattern programming exercises
│   ├── README.md                   # Python learning guide
│   └── slides/                     # Learning slides and materials
│

├── ml/                             # Machine Learning
│   ├── supervised_learning/
│   │   ├── 01-linear_regression/
│   │   ├── 02-logistic_regression/
│   │   ├── 03-decision_tree/
│   │   ├── 04-random_forest/
│   │   ├── 05-naive_bayes/
│   │   ├── 06-svm/
│   │   ├── 07-knn/
│   │   ├── 08-gradient_boosting/
│   │   └── 09-adaboost/
│   ├── unsupervised_learning/
│   │   ├── 01-k_means_clustering/
│   │   └── 02-pca/
│   └── README.md                   # ML learning guide
│
├── deep_learning/                  # Deep Learning with PyTorch
│   ├── 01-neural-network/
│   │   └── neural-network.py
│   ├── 02-convolutional-neural-network/
│   │   ├── cnn.py
│   │   └── data/MNIST/             # MNIST dataset
│   └── 03-recurrent-neural/
│       └── rnn.py
│
├── dsa/                            # Data Structures & Algorithms
│   ├── data_structures/
│   │   ├── 00_arrays_and_strings.py
│   │   ├── 01_linked_list.py
│   │   ├── 02_stacks.py
│   │   ├── 03_queue.py
│   │   ├── 04_hash_tables.py
│   │   ├── 05_heap.py
│   │   └── reverse_str.py
│   ├── algorithms/
│   │   ├── sorting/
│   │   ├── searching/
│   │   └── graph/
│   └── README.md                   # DSA learning guide
│
├── sql/                            # SQL queries and examples
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
│   ├── advanced/
│   │   ├── 01_ctes.sql
│   │   ├── 02_temp_tables.sql
│   │   ├── 03_stored_procedures.sql
│   │   └── 04_triggers_and_events.sql
│   └── README.md                   # SQL learning guide
│
├── statistics/                     # Statistical concepts
│   ├── normaliation-vs-standardization.py
│   └── statistics.md
│
├── web_scraping/                   # Web scraping examples
│   ├── 00-request_module.py
│   └── 01-beautifulsoup.py
│
├── javascript/                     # JavaScript fundamentals
│   └── javascript.js
│
├── linux/                          # Linux administration and commands
│   └── arch-pacman.md              # Arch Linux Pacman cheat sheet
│
├── projects/                       # Complete data science projects
│   ├── Data_sicence_project_workflow.py
│   └── README.md                   # Projects guide
│
├── data/                           # Centralized sample datasets
│   ├── chennai_cyclones.csv
│   ├── fish_dataset.csv
│   ├── my_data.csv
│   ├── public_holidays_dataset.csv
│   └── README.md                   # Dataset documentation
│
├── docker/                         # Docker resources
│   └── README.md
│
├── pdf's/                          # Reference materials
│   └── An Introduction to Statistical Learning.pdf
│
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Python dependencies
├── uv.lock                         # Lock file for uv
└── LICENSE                         # MIT License
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
- **Neural Networks** - Building and training artificial neural networks
- **Convolutional Neural Networks (CNNs)** - Image processing and computer vision
- **Recurrent Neural Networks (RNNs)** - Sequence modeling and time series
- **PyTorch** - Deep learning framework (torch, torchvision, torchaudio)

### Additional Technologies
- **SQL** - Database queries and data manipulation
- **Statistics** - Statistical concepts and analysis
- **Web Scraping** - Data extraction from websites
- **Data Structures & Algorithms** - Problem-solving fundamentals
- **Linux** - System administration and command-line tools (Arch Linux/Pacman)

## 💻 Usage

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
uv python install 3.12 # or uv python install 3.12+
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
- **Machine Learning**: scikit-learn
- **Deep Learning**: torch, torchvision, torchaudio (PyTorch ecosystem)
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

### Recommended Learning Order

1. **Python Fundamentals** (`python/` directory)
   - Start with data types, control flow, and functions
   - Progress to OOP, data structures, and exception handling
   - Master data science libraries (NumPy, Pandas, Matplotlib, Seaborn)
   - Explore advanced topics (multiprocessing, multithreading, decorators, generators)

2. **Data Science Tools** (`python/` notebooks 06-09)
   - NumPy for numerical computing
   - Pandas for data manipulation
   - Matplotlib and Seaborn for visualization

3. **Statistics** (`statistics/` directory)
   - Normalization vs standardization
   - Statistical concepts and analysis

4. **SQL** (`sql/` directory)
   - Begin with basic SELECT statements
   - Progress through joins, subqueries, and window functions
   - Master advanced topics like CTEs and stored procedures

5. **Machine Learning** (`ml/` directory)
   - Start with supervised learning (linear regression, logistic regression)
   - Explore tree-based methods (decision trees, random forest)
   - Learn ensemble methods (gradient boosting, AdaBoost)
   - Dive into unsupervised learning (K-means, PCA)

6. **Deep Learning** (`deep_learning/` directory)
   - Begin with neural networks fundamentals
   - Learn convolutional neural networks (CNNs)
   - Explore recurrent neural networks (RNNs)

7. **Data Structures & Algorithms** (`dsa/` directory)
   - Implement fundamental data structures
   - Practice sorting and searching algorithms
   - Solve graph problems

8. **Projects** (`projects/` directory)
   - Apply all concepts in end-to-end projects
   - Follow the data science workflow notebook

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for a detailed learning roadmap with time estimates.

## 📚 Documentation

This repository includes comprehensive documentation:

- **[README.md](README.md)** - Main overview and getting started guide (this file)
- **Directory READMEs** - Each major directory has its own README with specific guidance:
  - [Python README](python/README.md) - Python fundamentals guide
  - [ML README](ml/README.md) - Machine learning algorithms overview
  - [DSA README](dsa/README.md) - Data structures and algorithms guide
  - [SQL README](sql/README.md) - SQL learning path
  - [Projects README](projects/README.md) - Project workflow guide
  - [Data README](data/README.md) - Dataset documentation

## 🔍 Code Quality

This repository maintains good code quality standards with the following tools:

### Code Quality Tools

- **Black** - Code formatting for consistent style
- **Flake8** - Linting for code quality and PEP 8 compliance
- **MyPy** - Static type checking
- **Pytest** - Testing framework

### Quick Quality Check
```bash
# Install development tools
uv sync --group dev

# Format code
uv run black .

# Lint code
uv run flake8 .

# Type check
uv run mypy .

# Run tests (when available)
uv run pytest tests/
```

### Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose
- Comment complex logic and algorithms

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
- Add more examples and use cases
- Improve code comments and docstrings

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

## 🔗 Quick Links

### Directory Guides
- 📖 [Python Fundamentals](python/README.md) - Python learning guide
- 🤖 [Machine Learning](ml/README.md) - ML algorithms overview
- 🧮 [Data Structures & Algorithms](dsa/README.md) - DSA implementation guide
- 🗄️ [SQL Learning](sql/README.md) - SQL tutorials and examples
- 📊 [Projects](projects/README.md) - Data science project workflows

### Key Resources
- 📚 [Statistics](statistics/) - Statistical concepts and analysis
- 🕷️ [Web Scraping](web_scraping/) - Data extraction examples
- 🐧 [Linux](linux/) - Linux administration guides
- 🐳 [Docker](docker/) - Docker resources

---

## 📝 Notes

- All Jupyter notebooks are organized by topic and difficulty level
- Sample datasets are centralized in the `data/` directory
- Each major directory contains its own README with specific guidance
- Code examples are well-commented and include explanations
- Projects follow a standard data science workflow
