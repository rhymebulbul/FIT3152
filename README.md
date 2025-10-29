# FIT3152 Data Analytics

**Classification**: Machine Learning & Data Science

**Note**: This project was completed in R using R Markdown (.Rmd) for reproducible analysis and reporting.


## Assignments Overview

### Assignment 1: COVID-19 Behavioral Analysis
- **Focus**: Descriptive statistics and data visualization
- **Dataset**: PsyCorona baseline study (40,000 sample subset)
- **Analysis**: Comparative analysis of Croatia vs. other countries on pro-social behaviors during early pandemic
- **Techniques**: Data preprocessing, exploratory data analysis, ggplot2 visualization
- **Tools**: R, ggplot2, dplyr, tidyr

### Assignment 2: Phishing Website Detection
- **Focus**: Classification algorithm comparison
- **Dataset**: Phishing website data (2,000 sample subset, 25 features)
- **Models Implemented**:
  - Decision Trees
  - Naive Bayes
  - Bagging
  - Boosting (AdaBoost)
  - Random Forest
- **Evaluation**: 70/30 train-test split with accuracy metrics and ROC analysis
- **Tools**: R, caret, tree, e1071, adabag, randomForest, ROCR

### Assignment 3: Text Mining and Clustering
- **Focus**: Document clustering using hierarchical methods
- **Corpus**: 20 text documents
  - 5 Linux documentation articles
  - 5 Fast & Furious movie reviews
  - 5 Pirates of the Caribbean reviews
  - 5 Mission Impossible reviews
- **Techniques**:
  - Text preprocessing (tokenization, stemming, stopword removal)
  - Document-Term Matrix construction (10% sparsity threshold)
  - Hierarchical clustering (Euclidean and cosine distance)
- **Tools**: R, tm, cluster, igraph

## Repository Structure

```
FIT3152/
├── A1/                          # Assignment 1
│   ├── 3152.Rmd                # R Markdown source
│   ├── *.pdf                   # Final report
│   └── PsyCoronaBaselineExtract.csv
├── A2/                          # Assignment 2
│   ├── *.Rmd                   # R Markdown source
│   ├── *.pdf                   # Final report
│   └── PhishingData.csv
├── A3/                          # Assignment 3
│   ├── *.Rmd                   # R Markdown source
│   ├── *.pdf                   # Final report
│   └── text/                   # 20 text documents for clustering
└── README.md
```

## Technologies Used

- **Language**: R (via R Markdown)
- **Statistical Analysis**: Base R, dplyr
- **Visualization**: ggplot2
- **Machine Learning**: tree, e1071, randomForest, adabag, lightgbm, kernlab
- **Text Mining**: tm (text mining package)
- **Clustering**: cluster, igraph
- **Evaluation**: ROCR, caret

## Running the Code

### Prerequisites
```r
install.packages(c("ggplot2", "dplyr", "tidyr", "caret", "tree",
                   "e1071", "adabag", "randomForest", "ROCR",
                   "lightgbm", "kernlab", "tm", "cluster", "igraph"))
```

### Rendering Reports
```bash
# From RStudio or R console
Rscript -e "rmarkdown::render('A1/3152.Rmd')"
Rscript -e "rmarkdown::render('A2/<filename>.Rmd')"
Rscript -e "rmarkdown::render('A3/<filename>.Rmd')"
```

## Key Features

- **Reproducibility**: All analyses use seeded random sampling for consistent results
- **Complete Pipeline**: Data preprocessing → Model training → Evaluation → Reporting
- **Multiple Paradigms**: Statistical analysis, supervised learning, unsupervised learning, text mining
- **Professional Documentation**: Full R Markdown reports with code, analysis, and visualizations

## Python Package Equivalents

For reference, here are the Python equivalents of the R packages used in this project:

### Assignment 1 (COVID-19 Analysis)

| R Package | Python Equivalent | Purpose |
|-----------|------------------|---------|
| `ggplot2` | `matplotlib`, `seaborn`, or `plotnine` | Visualization (plotnine is most similar to ggplot2) |
| `dplyr` | `pandas` | Data manipulation |
| `tidyr` | `pandas` | Data tidying/reshaping |

### Assignment 2 (Phishing Detection)

| R Package | Python Equivalent | Purpose |
|-----------|------------------|---------|
| `dplyr` | `pandas` | Data preprocessing |
| `caret` | `scikit-learn` | ML pipeline and preprocessing |
| `tree` | `scikit-learn.tree` (DecisionTreeClassifier) | Decision trees |
| `e1071` | `scikit-learn.svm` | SVM and Naive Bayes |
| `adabag` | `scikit-learn.ensemble` (AdaBoostClassifier) | AdaBoost |
| `randomForest` | `scikit-learn.ensemble` (RandomForestClassifier) | Random forests |
| `ROCR` | `scikit-learn.metrics` | ROC curves and model evaluation |
| `lightgbm` | `lightgbm` | Gradient boosting (same package name!) |
| `kernlab` | `scikit-learn.svm` (with kernel options) | Kernel-based ML methods |

### Assignment 3 (Text Mining)

| R Package | Python Equivalent | Purpose |
|-----------|------------------|---------|
| `tm` | `nltk`, `sklearn.feature_extraction.text`, or `gensim` | Text mining and corpus management |
| `cluster` | `scikit-learn.cluster` | Clustering algorithms |
| `igraph` | `networkx` or `igraph` (Python version) | Network analysis |

### Core Python Stack Summary

For all three assignments combined, the core Python packages would be:
- **`pandas`** - replaces dplyr/tidyr for data manipulation
- **`scikit-learn`** - replaces most ML packages (caret, tree, e1071, adabag, randomForest, ROCR, kernlab, cluster)
- **`matplotlib`/`seaborn`/`plotnine`** - replaces ggplot2
- **`nltk`** or **`sklearn.feature_extraction.text`** - replaces tm
- **`lightgbm`** - same in both R and Python
- **`networkx`** or **`igraph`** - replaces igraph
