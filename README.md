# FIT3152 Data Analytics - Coursework

Academic assignments completed for **FIT3152 Data Analytics** at Monash University (Semester 1, 2024).

> **Note**: This repository contains university coursework assignments, not independent research projects. The work demonstrates applied data analytics techniques as taught in the course.

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

## Academic Context

**Course**: FIT3152 Data Analytics
**Institution**: Monash University
**Semester**: Semester 1, 2024