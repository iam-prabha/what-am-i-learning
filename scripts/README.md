# Utility Scripts

Helper scripts and automation tools for the repository.

## 📁 Contents

Add utility scripts here for:

- **Data Processing** - Scripts to clean, transform, or generate data
- **Automation** - Workflow automation and batch processing
- **Analysis Tools** - Helper functions for data analysis
- **Setup Scripts** - Environment setup and configuration
- **Testing** - Test utilities and validation scripts

## 💡 Example Scripts

```python
# scripts/load_data.py
"""Helper to load common datasets"""
import pandas as pd
from pathlib import Path

def load_dataset(name):
    data_dir = Path(__file__).parent.parent / 'data'
    return pd.read_csv(data_dir / f'{name}.csv')
```

```python
# scripts/model_utils.py
"""Common ML model utilities"""
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
```

## 🎯 Best Practices

- Use descriptive filenames (e.g., `train_model.py`, not `script1.py`)
- Add docstrings and comments
- Make scripts reusable across projects
- Include error handling
- Log important operations

## 🔧 Usage

```bash
# Run a script
python scripts/data_processing.py

# Import in notebooks
from scripts.model_utils import split_data
```
