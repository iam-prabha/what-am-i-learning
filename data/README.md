# Sample Datasets

Centralized location for all sample datasets used throughout the learning materials.

## 📊 Available Datasets

### Chennai Cyclones (chennai_cyclones.csv)
- **Description**: Historical cyclone data for Chennai region
- **Use Cases**: Time series analysis, weather prediction
- **Used In**: Python notebooks, data visualization examples

### Fish Dataset (fish_dataset.csv)
- **Description**: Fish species measurements and characteristics
- **Use Cases**: Classification, regression analysis
- **Used In**: Machine learning examples

### My Data (my_data.csv)
- **Description**: General purpose sample data
- **Use Cases**: Data manipulation practice
- **Used In**: Python/Pandas tutorials

### Public Holidays Dataset (public_holidays_dataset.csv)
- **Description**: Public holiday information
- **Use Cases**: Date/time analysis, calendar operations
- **Used In**: Pandas date handling examples

## 🔗 Dataset Sources

Datasets are referenced across multiple directories:
- `../python/` - Data manipulation examples
- `../ml/` - Machine learning training
- `../statistics/` - Statistical analysis
- `../projects/` - Complete project workflows

## 📝 Usage in Notebooks

When loading data from notebooks, use relative paths:

```python
import pandas as pd

# From python directory
df = pd.read_csv('../data/fish_dataset.csv')

# From ml directory
df = pd.read_csv('../../data/chennai_cyclones.csv')

# From project directory
df = pd.read_csv('../data/public_holidays_dataset.csv')
```

## ➕ Adding New Datasets

When adding new datasets:
1. Place CSV/data files in this directory
2. Use lowercase with underscores for filenames
3. Document the dataset above
4. Update references in notebooks
5. Commit with descriptive message

## 🔒 Data Privacy

- Never commit sensitive or personal data
- Use sample/synthetic data for learning
- Check `.gitignore` for excluded data patterns
