"""
FIT3152 - Data Analytics
Assignment 1, Semester 1, 2024
Python Translation

Analysis of country-level predictors of pro-social behaviours to reduce
the spread of COVID-19 during the early stages of the pandemic

Student ID: 42857193
Focus Country: Croatia
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Set random seed for reproducibility
STUDENT_ID = 42857193
np.random.seed(STUDENT_ID)

# ============================================================================
# TASK 1: Descriptive Analysis and Pre-processing
# ============================================================================

def load_and_sample_data(filepath, n_samples=40000, seed=STUDENT_ID):
    """
    Load the PsyCorona dataset and take a random sample.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    n_samples : int
        Number of samples to draw
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Sampled dataframe
    """
    print("Loading data...")
    cvbase = pd.read_csv(filepath)

    # Sample 40,000 rows
    cvbase = cvbase.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"Data loaded: {cvbase.shape[0]} rows, {cvbase.shape[1]} columns")
    return cvbase


def descriptive_analysis(df):
    """
    Perform descriptive analysis on the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    """
    print("\n" + "="*80)
    print("TASK 1(a): Descriptive Analysis")
    print("="*80)

    # Dimensions
    print(f"\nDataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

    # Data types
    print("\nData types:")
    print(df.dtypes.value_counts())

    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Text attributes
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nText attributes: {text_cols}")

    # Missing values
    print("\nMissing values by column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

    # Country analysis
    print("\nCountry Analysis:")
    country_counts = df['coded_country'].value_counts()
    print(f"Number of unique countries: {df['coded_country'].nunique()}")
    print(f"Country with most responses: {country_counts.index[0]} ({country_counts.iloc[0]} responses)")
    print(f"Country with least responses: {country_counts.index[-1]} ({country_counts.iloc[-1]} responses)")

    if 'Croatia' in country_counts.index:
        print(f"Croatia responses: {country_counts['Croatia']}")

    # Age analysis
    if 'age' in df.columns:
        print(f"\nMean age group: {df['age'].mean():.3f}")
        print("This suggests majority of participants are aged 35-44 years")

    return missing_df


def preprocess_data(df):
    """
    Preprocess the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to preprocess

    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    print("\n" + "="*80)
    print("TASK 1(b): Data Preprocessing")
    print("="*80)

    df_processed = df.copy()

    # Replace NaN values with 0 (especially for employstatus columns)
    print("\nReplacing missing values with 0...")
    df_processed = df_processed.fillna(0)

    print(f"Missing values after preprocessing: {df_processed.isnull().sum().sum()}")

    return df_processed


# ============================================================================
# TASK 2: Focus Country vs All Other Countries
# ============================================================================

def split_countries(df, focus_country='Croatia'):
    """
    Split data into focus country and others.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    focus_country : str
        The focus country name

    Returns:
    --------
    tuple
        (croatia_df, others_df)
    """
    croatia = df[df['coded_country'] == focus_country].copy()
    others = df[df['coded_country'] != focus_country].copy()

    print(f"\n{focus_country} data: {len(croatia)} rows")
    print(f"Other countries data: {len(others)} rows")

    return croatia, others


def plot_mean_responses(df, title, color='purple', figsize=(12, 10)):
    """
    Create bar chart of mean responses for each numeric column.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    title : str
        Plot title
    color : str
        Bar color
    figsize : tuple
        Figure size
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate means
    means = df[numeric_cols].mean()

    # Sort by mean value for better visualization
    means = means.sort_values()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    means.plot(kind='barh', ax=ax, color=color)
    ax.set_xlabel('Mean responses')
    ax.set_ylabel('Survey questions')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def compare_countries_means(croatia, others):
    """
    Compare mean responses between Croatia and other countries.
    """
    print("\n" + "="*80)
    print("TASK 2(a): Focus Country vs All Others - Mean Comparison")
    print("="*80)

    # Plot for Croatia
    fig1 = plot_mean_responses(
        croatia,
        'Mean of responses for each question in Croatia',
        color='purple'
    )
    plt.savefig('croatia_means.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot for others
    fig2 = plot_mean_responses(
        others,
        'Mean of responses for each question over the globe',
        color='lightblue'
    )
    plt.savefig('world_means.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nPlots saved: croatia_means.png, world_means.png")

    # Key differences
    numeric_cols = croatia.select_dtypes(include=[np.number]).columns
    croatia_means = croatia[numeric_cols].mean()
    others_means = others[numeric_cols].mean()

    differences = croatia_means - others_means
    print("\nTop 10 largest differences (Croatia - World):")
    print(differences.abs().nlargest(10))


def plot_correlation_heatmap(df, title, figsize=(12, 10)):
    """
    Create correlation heatmap.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=1.0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'correlation'}
    )
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig, corr_matrix


def fit_linear_models(df, focus_country_name='Croatia'):
    """
    Fit linear regression models for pro-social attitudes.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    focus_country_name : str
        Name of the focus country for reporting

    Returns:
    --------
    dict
        Dictionary containing model results
    """
    print("\n" + "="*80)
    print(f"TASK 2(b): Predicting Pro-social Attitudes for {focus_country_name}")
    print("="*80)

    # Prepare data - exclude non-predictive columns
    exclude_cols = ['coded_country']
    prosocial_vars = ['c19ProSo01', 'c19ProSo02', 'c19ProSo03', 'c19ProSo04']

    results = {}

    for target in prosocial_vars:
        print(f"\n{target}")
        print("-" * 40)

        # Prepare features and target
        feature_cols = [col for col in df.columns
                       if col not in exclude_cols + prosocial_vars and col != target]

        # Ensure only numeric columns
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target]

        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) == 0:
            print("No valid data for modeling")
            continue

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_clean, y_clean)

        # Calculate R-squared
        r_squared = model.score(X_clean, y_clean)

        # Calculate adjusted R-squared
        n = len(y_clean)
        p = X_clean.shape[1]
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        print(f"R-squared: {r_squared:.6f}")
        print(f"Adjusted R-squared: {adj_r_squared:.6f}")

        # Get coefficients and p-values
        # For p-values, we need to calculate t-statistics
        predictions = model.predict(X_clean)
        residuals = y_clean - predictions
        mse = np.sum(residuals**2) / (n - p - 1)

        # Calculate standard errors (handle singular matrix with pseudo-inverse)
        try:
            # Try regular inverse
            XTX_inv = np.linalg.inv(X_clean.T.dot(X_clean))
            var_coef = mse * XTX_inv.diagonal()
        except np.linalg.LinAlgError:
            # Use Moore-Penrose pseudo-inverse if matrix is singular
            XTX_pinv = np.linalg.pinv(X_clean.T.dot(X_clean))
            var_coef = mse * XTX_pinv.diagonal()

        std_errors = np.sqrt(np.abs(var_coef))  # abs to handle numerical errors

        # T-statistics (avoid division by zero)
        t_stats = np.divide(model.coef_, std_errors,
                            out=np.zeros_like(model.coef_),
                            where=std_errors!=0)

        # P-values (two-tailed test)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

        # Find significant predictors (p < 0.001)
        significant_mask = p_values < 0.001
        significant_features = X_clean.columns[significant_mask].tolist()
        significant_coefs = model.coef_[significant_mask]

        print(f"99.9% confidence interval significant predictors:")
        if len(significant_features) > 0:
            for feat, coef in zip(significant_features, significant_coefs):
                print(f"  {feat}: {coef:.6f}")
        else:
            print("  None")

        # Store results
        results[target] = {
            'model': model,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'significant_features': significant_features,
            'significant_coefs': significant_coefs,
            'all_features': X_clean.columns.tolist(),
            'all_coefs': model.coef_
        }

    return results


def create_predictor_table(croatia_results, others_results, similar_results=None):
    """
    Create a table showing significant predictors for each model.

    Parameters:
    -----------
    croatia_results : dict
        Results from Croatia models
    others_results : dict
        Results from other countries models
    similar_results : dict, optional
        Results from similar countries models
    """
    print("\n" + "="*80)
    print("Significant Predictors Summary")
    print("="*80)

    # Collect all predictors
    all_predictors = set()
    all_models = []

    # Croatia
    for target, results in croatia_results.items():
        model_name = f"Croatia_{target}"
        all_models.append(model_name)
        all_predictors.update(results['significant_features'])

    # Others
    for target, results in others_results.items():
        model_name = f"RoW_{target}"
        all_models.append(model_name)
        all_predictors.update(results['significant_features'])

    # Similar countries
    if similar_results:
        for target, results in similar_results.items():
            model_name = f"Similar_{target}"
            all_models.append(model_name)
            all_predictors.update(results['significant_features'])

    # Create table
    predictor_table = pd.DataFrame(0, index=sorted(all_predictors), columns=all_models)

    # Fill in the table
    for target, results in croatia_results.items():
        model_name = f"Croatia_{target}"
        for feat in results['significant_features']:
            predictor_table.loc[feat, model_name] = 1

    for target, results in others_results.items():
        model_name = f"RoW_{target}"
        for feat in results['significant_features']:
            predictor_table.loc[feat, model_name] = 1

    if similar_results:
        for target, results in similar_results.items():
            model_name = f"Similar_{target}"
            for feat in results['significant_features']:
                predictor_table.loc[feat, model_name] = 1

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        predictor_table,
        cmap=['lightgray', 'green'],
        cbar=False,
        linewidths=0.5,
        linecolor='black',
        square=True,
        ax=ax
    )
    ax.set_title('Significant predictors for individual models')
    ax.set_xlabel('Models')
    ax.set_ylabel('Predictors')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig, predictor_table


# ============================================================================
# TASK 3: Focus Country vs Cluster of Similar Countries
# ============================================================================

def load_clustering_data(filepath):
    """
    Load external data for clustering.

    Parameters:
    -----------
    filepath : str
        Path to the clustering data CSV

    Returns:
    --------
    pd.DataFrame
        Clustering data
    """
    print("\n" + "="*80)
    print("TASK 3(a): Clustering Similar Countries")
    print("="*80)

    external = pd.read_csv(filepath)
    print(f"\nLoaded external data: {external.shape}")
    print(f"Columns: {external.columns.tolist()}")

    return external


def perform_clustering(external_df, n_clusters=None, random_state=STUDENT_ID):
    """
    Perform K-means clustering on external data.

    Parameters:
    -----------
    external_df : pd.DataFrame
        External data with clustering indicators
    n_clusters : int, optional
        Number of clusters. If None, will use n_countries/5
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        (clusters_df, kmeans_model, similar_countries)
    """
    # Remove countries with NA values
    cleaned_external = external_df.dropna()
    print(f"\nCountries after removing NAs: {len(cleaned_external)}")

    # Select numeric columns for clustering
    numeric_cols = cleaned_external.select_dtypes(include=[np.number]).columns
    X = cleaned_external[numeric_cols]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = round(len(cleaned_external) / 5)

    print(f"Performing K-means clustering with {n_clusters} clusters...")

    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=15, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    # Create results dataframe
    clusters_df = pd.DataFrame({
        'country': cleaned_external['country'].values,
        'cluster': clusters
    })

    # Find Croatia's cluster
    croatia_cluster = clusters_df[clusters_df['country'] == 'Croatia']['cluster'].values

    if len(croatia_cluster) > 0:
        target_cluster = croatia_cluster[0]
        similar_countries = clusters_df[clusters_df['cluster'] == target_cluster]

        print(f"\nCroatia is in cluster {target_cluster}")
        print(f"Similar countries:")
        print(similar_countries)
    else:
        print("\nCroatia not found in clustering data")
        similar_countries = None

    return clusters_df, kmeans, similar_countries


def analyze_similar_countries(cvbase, similar_countries_df):
    """
    Analyze the cluster of similar countries.

    Parameters:
    -----------
    cvbase : pd.DataFrame
        Main dataset
    similar_countries_df : pd.DataFrame
        DataFrame with similar countries

    Returns:
    --------
    pd.DataFrame
        Data for similar countries (excluding Croatia)
    """
    print("\n" + "="*80)
    print("TASK 3(b): Analyzing Similar Countries Cluster")
    print("="*80)

    # Merge with baseline data
    similar_country_list = similar_countries_df['country'].tolist()
    clustered = cvbase[cvbase['coded_country'].isin(similar_country_list)].copy()

    # Exclude Croatia
    clustered = clustered[clustered['coded_country'] != 'Croatia']

    print(f"\nSimilar countries data (excl. Croatia): {len(clustered)} rows")
    print(f"Countries: {clustered['coded_country'].unique().tolist()}")

    return clustered


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    """
    Main analysis pipeline.
    """
    print("="*80)
    print("FIT3152 Assignment 1 - Python Translation")
    print("COVID-19 Pro-social Behaviors Analysis")
    print("="*80)

    # Load data (from parent directory)
    cvbase = load_and_sample_data('../PsyCoronaBaselineExtract.csv')

    # Task 1: Descriptive Analysis
    missing_df = descriptive_analysis(cvbase)
    cvbase_processed = preprocess_data(cvbase)

    # Task 2: Focus Country vs All Others
    croatia, others = split_countries(cvbase_processed, 'Croatia')

    # 2(a): Compare means
    compare_countries_means(croatia, others)

    # 2(b): Correlation and modeling for Croatia
    print("\n" + "="*80)
    print("Correlation Analysis - Croatia")
    print("="*80)
    fig_corr_croatia, corr_croatia = plot_correlation_heatmap(
        croatia,
        "Correlation between each of Croatia's predictors"
    )
    plt.savefig('croatia_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    croatia_results = fit_linear_models(croatia, 'Croatia')

    # 2(c): Correlation and modeling for others
    print("\n" + "="*80)
    print("Correlation Analysis - Rest of World")
    print("="*80)
    fig_corr_others, corr_others = plot_correlation_heatmap(
        others,
        "Correlation between each of the world's predictors"
    )
    plt.savefig('world_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    others_results = fit_linear_models(others, 'Rest of World')

    # Task 3: Clustering
    try:
        external = load_clustering_data('../task3.csv')
        clusters_df, kmeans_model, similar_countries = perform_clustering(external)

        if similar_countries is not None:
            clustered = analyze_similar_countries(cvbase_processed, similar_countries)

            # Correlation for similar countries
            fig_corr_similar, corr_similar = plot_correlation_heatmap(
                clustered,
                "Correlation between predictors for countries similar to Croatia"
            )
            plt.savefig('similar_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Model similar countries
            similar_results = fit_linear_models(clustered, 'Similar Countries')

            # Create predictor comparison table
            fig_table, predictor_table = create_predictor_table(
                croatia_results,
                others_results,
                similar_results
            )
            plt.savefig('predictor_table.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("\nAnalysis complete!")
            print("Generated files:")
            print("  - croatia_means.png")
            print("  - world_means.png")
            print("  - croatia_correlation.png")
            print("  - world_correlation.png")
            print("  - similar_correlation.png")
            print("  - predictor_table.png")

    except FileNotFoundError:
        print("\nWarning: task3.csv not found. Skipping clustering analysis.")
        print("Generating predictor table for Croatia and Rest of World only...")

        fig_table, predictor_table = create_predictor_table(
            croatia_results,
            others_results
        )
        plt.savefig('predictor_table.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
