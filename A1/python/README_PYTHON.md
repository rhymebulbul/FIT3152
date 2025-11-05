# FIT3152 Assignment 1 - Python Translation

This directory contains a complete Python translation of Assignment 1 from R to Python.

## Files

1. **assignment1_python.py** - Complete Python script with all analysis functions
2. **Assignment1_Python.ipynb** - Jupyter notebook version for interactive analysis
3. **README_PYTHON.md** - This file

## Requirements

### Python Packages

Install required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Or using conda:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Data Files

Ensure the following data files are in the parent directory (A1/):
- `../PsyCoronaBaselineExtract.csv` - Main survey data
- `../task3.csv` - External indicators for clustering (for Task 3)

**Note:** The Python scripts have been updated to look for data files in the parent directory.

## Usage

### Option 1: Run the complete script

```bash
cd A1/python
source venv/bin/activate
python assignment1_python.py
```

This will:
1. Load and preprocess the data
2. Perform descriptive analysis
3. Compare Croatia to other countries
4. Fit linear regression models
5. Perform clustering analysis
6. Generate all visualizations

### Option 2: Use the Jupyter Notebook

```bash
jupyter notebook Assignment1_Python.ipynb
```

Then run cells sequentially to perform the analysis interactively.

### Option 3: Import as a module

```python
import assignment1_python as a1

# Load data
cvbase = a1.load_and_sample_data('PsyCoronaBaselineExtract.csv')

# Descriptive analysis
missing_df = a1.descriptive_analysis(cvbase)

# Preprocess
cvbase = a1.preprocess_data(cvbase)

# Split countries
croatia, others = a1.split_countries(cvbase, 'Croatia')

# Fit models
croatia_results = a1.fit_linear_models(croatia, 'Croatia')
```

## Key Differences from R Implementation

### Libraries
| R | Python |
|---|--------|
| `ggplot2` | `matplotlib` + `seaborn` |
| `dplyr` | `pandas` |
| `tidyr` | `pandas` |
| Base R stats | `scipy.stats` |
| - | `scikit-learn` (for clustering and regression) |

### Syntax Differences

**Data Sampling:**
```r
# R
cvbase <- cvbase[sample(nrow(cvbase), 40000), ]
```
```python
# Python
cvbase = cvbase.sample(n=40000, random_state=STUDENT_ID)
```

**Missing Value Replacement:**
```r
# R
cvbase[is.na(cvbase)] <- 0
```
```python
# Python
cvbase = cvbase.fillna(0)
```

**Subsetting Data:**
```r
# R
croatia <- cvbase[cvbase$coded_country == "Croatia", ]
```
```python
# Python
croatia = cvbase[cvbase['coded_country'] == 'Croatia']
```

**Linear Regression:**
```r
# R
model <- lm(c19ProSo01 ~ ., data=subset(...))
summary(model)
```
```python
# Python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
r_squared = model.score(X, y)
```

**K-means Clustering:**
```r
# R
kfit <- kmeans(cleaned_external[, 2:9], k, nstart=15)
```
```python
# Python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k, n_init=15, random_state=seed)
clusters = kmeans.fit_predict(scaled_data)
```

## Output Files

The script generates the following visualizations:
- `croatia_means.png` - Mean responses for Croatia
- `world_means.png` - Mean responses for all other countries
- `croatia_correlation.png` - Correlation heatmap for Croatia
- `world_correlation.png` - Correlation heatmap for other countries
- `similar_correlation.png` - Correlation heatmap for similar countries
- `predictor_table.png` - Comparison of significant predictors

## Student Information

- **Student ID:** 42857193
- **Focus Country:** Croatia
- **Random Seed:** 42857193 (ensures reproducibility)

## Functions Overview

### Data Loading and Preprocessing
- `load_and_sample_data()` - Load CSV and sample 40,000 rows
- `descriptive_analysis()` - Generate descriptive statistics
- `preprocess_data()` - Handle missing values

### Country Analysis
- `split_countries()` - Separate focus country from others
- `plot_mean_responses()` - Create bar charts of means
- `compare_countries_means()` - Compare focus vs others
- `plot_correlation_heatmap()` - Create correlation heatmaps

### Modeling
- `fit_linear_models()` - Fit linear regression for pro-social attitudes
- `create_predictor_table()` - Summarize significant predictors

### Clustering
- `load_clustering_data()` - Load external indicators
- `perform_clustering()` - K-means clustering
- `analyze_similar_countries()` - Extract similar country data

### Main Pipeline
- `main()` - Run complete analysis pipeline

## Notes

1. **Reproducibility:** The random seed is set to 42857193 (student ID)
2. **Statistical Tests:** P-values are calculated manually using t-statistics (matching R's lm output)
3. **Visualizations:** Uses seaborn style to approximate ggplot2 aesthetics
4. **Performance:** Python's pandas and numpy provide efficient data manipulation similar to R's tidyverse
5. **Singular Matrix Handling:** The linear regression uses pseudo-inverse when the matrix is singular due to multicollinearity

## Troubleshooting

**Issue:** `FileNotFoundError` for CSV files
- **Solution:** Ensure data files are in the same directory or provide full path

**Issue:** Missing packages
- **Solution:** Install required packages: `pip install pandas numpy matplotlib seaborn scikit-learn scipy`

**Issue:** Memory errors with large dataset
- **Solution:** The script processes 40,000 rows. Reduce if needed in `load_and_sample_data()`

**Issue:** Figures not displaying
- **Solution:** In Jupyter, use `%matplotlib inline`. In scripts, add `plt.show()`

## Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Original R Assignment](3152.Rmd)

## License

This is a translation of academic coursework for FIT3152 at Monash University.
