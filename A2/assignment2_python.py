"""
FIT3152 Assignment 2 - Phishing Website Detection
Python Translation from R

Student ID: 42857193
Dataset: PhishingData.csv
Task: Comparative analysis of classification algorithms for phishing detection

This script implements 12 questions covering:
- Data exploration and preprocessing
- Multiple classification models (Decision Tree, Naive Bayes, Ensemble methods, etc.)
- Model evaluation using accuracy, confusion matrices, and ROC curves
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
STUDENT_ID = 42857193
np.random.seed(STUDENT_ID)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(filepath='PhishingData.csv'):
    """
    Load the phishing dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the PhishingData.csv file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    print("Loading phishing dataset...")
    data = pd.read_csv(filepath)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def question1_dimensions(data):
    """
    Question 1: Report dimensions and first few rows of the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        The phishing dataset
    """
    print("\n" + "="*80)
    print("QUESTION 1: Dataset Dimensions and Preview")
    print("="*80)
    print(f"\nDataset dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nDataset structure:")
    print(data.dtypes)


def question2_sample_data(data):
    """
    Question 2: Sample data following the assignment specification.
    - Select 10 random values from 1-50
    - Filter data where A01 matches those values
    - Sample 2000 rows from filtered data

    Parameters:
    -----------
    data : pd.DataFrame
        The full phishing dataset

    Returns:
    --------
    pd.DataFrame
        Sampled dataset with 2000 rows
    """
    print("\n" + "="*80)
    print("QUESTION 2: Data Sampling")
    print("="*80)

    # Set seed for reproducibility
    np.random.seed(STUDENT_ID)

    # Create L: sample 10 values from 1-50
    L = np.random.choice(range(1, 51), size=10, replace=False)
    print(f"\n10 randomly selected values from 1-50: {sorted(L)}")

    # Filter data where A01 is in L
    Phish = data[data['A01'].isin(L)].copy()
    print(f"Rows after filtering by A01: {Phish.shape[0]}")

    # Sample 2000 rows
    PD = Phish.sample(n=2000, random_state=STUDENT_ID).reset_index(drop=True)
    print(f"Final sampled dataset: {PD.shape[0]} rows")

    return PD


def question3_missing_values(data):
    """
    Question 3: Check for missing values.

    Parameters:
    -----------
    data : pd.DataFrame
        The sampled dataset
    """
    print("\n" + "="*80)
    print("QUESTION 3: Missing Values Analysis")
    print("="*80)

    missing_counts = data.isnull().sum()
    missing_percent = (missing_counts / len(data)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing Percentage': missing_percent.values
    })

    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) == 0:
        print("\nNo missing values found in the dataset!")
    else:
        print("\nMissing values summary:")
        print(missing_df)

    return missing_df


def question4_handle_missing(data):
    """
    Question 4: Handle missing values by removing rows.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with potential missing values

    Returns:
    --------
    pd.DataFrame
        Dataset with missing values removed
    """
    print("\n" + "="*80)
    print("QUESTION 4: Handle Missing Values")
    print("="*80)

    initial_rows = len(data)
    data_clean = data.dropna().reset_index(drop=True)
    final_rows = len(data_clean)

    print(f"\nInitial rows: {initial_rows}")
    print(f"Rows after removing missing values: {final_rows}")
    print(f"Rows removed: {initial_rows - final_rows}")

    return data_clean


def question5_train_test_split(data):
    """
    Question 5: Split data into training (70%) and testing (30%) sets.

    Parameters:
    -----------
    data : pd.DataFrame
        Clean dataset

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*80)
    print("QUESTION 5: Train-Test Split")
    print("="*80)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Convert Class to integer if it's not already
    y = y.astype(int)

    # 70-30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=STUDENT_ID, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts())
    print(f"\nClass distribution in testing set:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def question6_decision_tree(X_train, X_test, y_train, y_test):
    """
    Question 6: Build and evaluate a Decision Tree classifier.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    tuple
        (model, accuracy, predictions)
    """
    print("\n" + "="*80)
    print("QUESTION 6: Decision Tree Classifier")
    print("="*80)

    # Build decision tree
    dt_model = DecisionTreeClassifier(random_state=STUDENT_ID)
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred = dt_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Decision Tree - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('A2/decision_tree_confusion.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'decision_tree_confusion.png'")
    plt.close()

    return dt_model, accuracy, y_pred


def question7_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Question 7: Build and evaluate a Naive Bayes classifier.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    tuple
        (model, accuracy, predictions)
    """
    print("\n" + "="*80)
    print("QUESTION 7: Naive Bayes Classifier")
    print("="*80)

    # Build Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = nb_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nNaive Bayes Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return nb_model, accuracy, y_pred


def question8_bagging(X_train, X_test, y_train, y_test):
    """
    Question 8: Build and evaluate a Bagging classifier.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    tuple
        (model, accuracy, predictions)
    """
    print("\n" + "="*80)
    print("QUESTION 8: Bagging Classifier")
    print("="*80)

    # Build Bagging model with Decision Tree as base estimator
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=STUDENT_ID),
        n_estimators=100,
        random_state=STUDENT_ID
    )
    bagging_model.fit(X_train, y_train)

    # Make predictions
    y_pred = bagging_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nBagging Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return bagging_model, accuracy, y_pred


def question9_boosting(X_train, X_test, y_train, y_test):
    """
    Question 9: Build and evaluate an AdaBoost (Boosting) classifier.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    tuple
        (model, accuracy, predictions)
    """
    print("\n" + "="*80)
    print("QUESTION 9: AdaBoost (Boosting) Classifier")
    print("="*80)

    # Build AdaBoost model
    boosting_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=STUDENT_ID),
        n_estimators=100,
        random_state=STUDENT_ID,
        algorithm='SAMME'
    )
    boosting_model.fit(X_train, y_train)

    # Make predictions
    y_pred = boosting_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAdaBoost Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return boosting_model, accuracy, y_pred


def question10_random_forest(X_train, X_test, y_train, y_test):
    """
    Question 10: Build and evaluate a Random Forest classifier.
    Also perform feature importance analysis.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    tuple
        (model, accuracy, predictions)
    """
    print("\n" + "="*80)
    print("QUESTION 10: Random Forest Classifier")
    print("="*80)

    # Build Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=STUDENT_ID
    )
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nRandom Forest Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Random Forest - Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('A2/random_forest_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'random_forest_importance.png'")
    plt.close()

    return rf_model, accuracy, y_pred


def question11_advanced_models(X_train, X_test, y_train, y_test):
    """
    Question 11: Build and evaluate advanced models:
    - Pruned Decision Tree (via max_depth)
    - Gradient-Boosted Trees (LightGBM)

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    dict
        Dictionary containing models, accuracies, and predictions
    """
    print("\n" + "="*80)
    print("QUESTION 11: Advanced Models")
    print("="*80)

    results = {}

    # 11a: Pruned Decision Tree
    print("\n11a. Pruned Decision Tree (max_depth=5)")
    print("-" * 40)
    pruned_dt = DecisionTreeClassifier(max_depth=5, random_state=STUDENT_ID)
    pruned_dt.fit(X_train, y_train)
    pruned_pred = pruned_dt.predict(X_test)
    pruned_acc = accuracy_score(y_test, pruned_pred)
    pruned_cm = confusion_matrix(y_test, pruned_pred)

    print(f"Pruned Decision Tree Accuracy: {pruned_acc:.4f}")
    print("\nConfusion Matrix:")
    print(pruned_cm)

    results['pruned_dt'] = {
        'model': pruned_dt,
        'accuracy': pruned_acc,
        'predictions': pruned_pred
    }

    # 11b: LightGBM (Gradient-Boosted Trees)
    print("\n11b. LightGBM (Gradient-Boosted Trees)")
    print("-" * 40)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=STUDENT_ID,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_cm = confusion_matrix(y_test, lgb_pred)

    print(f"LightGBM Accuracy: {lgb_acc:.4f}")
    print("\nConfusion Matrix:")
    print(lgb_cm)

    results['lightgbm'] = {
        'model': lgb_model,
        'accuracy': lgb_acc,
        'predictions': lgb_pred
    }

    return results


def question12_neural_network_svm(X_train, X_test, y_train, y_test):
    """
    Question 12: Build and evaluate:
    - Artificial Neural Network (MLP)
    - Support Vector Machine (SVM)

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels

    Returns:
    --------
    dict
        Dictionary containing models, accuracies, and predictions
    """
    print("\n" + "="*80)
    print("QUESTION 12: Neural Network and SVM")
    print("="*80)

    results = {}

    # Scale features for Neural Network and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 12a: Artificial Neural Network (MLP)
    print("\n12a. Artificial Neural Network (MLP)")
    print("-" * 40)
    ann_model = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        max_iter=1000,
        random_state=STUDENT_ID
    )
    ann_model.fit(X_train_scaled, y_train)
    ann_pred = ann_model.predict(X_test_scaled)
    ann_acc = accuracy_score(y_test, ann_pred)
    ann_cm = confusion_matrix(y_test, ann_pred)

    print(f"Neural Network Accuracy: {ann_acc:.4f}")
    print("\nConfusion Matrix:")
    print(ann_cm)

    results['ann'] = {
        'model': ann_model,
        'accuracy': ann_acc,
        'predictions': ann_pred,
        'scaler': scaler
    }

    # 12b: Support Vector Machine (SVM)
    print("\n12b. Support Vector Machine (RBF kernel)")
    print("-" * 40)
    svm_model = SVC(kernel='radial', probability=True, random_state=STUDENT_ID)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)

    print(f"SVM Accuracy: {svm_acc:.4f}")
    print("\nConfusion Matrix:")
    print(svm_cm)

    results['svm'] = {
        'model': svm_model,
        'accuracy': svm_acc,
        'predictions': svm_pred,
        'scaler': scaler
    }

    return results


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for all models.

    Parameters:
    -----------
    models_dict : dict
        Dictionary containing all trained models
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing labels
    """
    print("\n" + "="*80)
    print("ROC Curves for All Models")
    print("="*80)

    plt.figure(figsize=(12, 8))

    # Decision Tree
    if 'decision_tree' in models_dict:
        y_proba = models_dict['decision_tree']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {roc_auc:.3f})')

    # Naive Bayes
    if 'naive_bayes' in models_dict:
        y_proba = models_dict['naive_bayes']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {roc_auc:.3f})')

    # Bagging
    if 'bagging' in models_dict:
        y_proba = models_dict['bagging']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Bagging (AUC = {roc_auc:.3f})')

    # Boosting
    if 'boosting' in models_dict:
        y_proba = models_dict['boosting']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AdaBoost (AUC = {roc_auc:.3f})')

    # Random Forest
    if 'random_forest' in models_dict:
        y_proba = models_dict['random_forest']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')

    # Pruned Decision Tree
    if 'pruned_dt' in models_dict:
        y_proba = models_dict['pruned_dt']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Pruned DT (AUC = {roc_auc:.3f})')

    # LightGBM
    if 'lightgbm' in models_dict:
        y_proba = models_dict['lightgbm']['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'LightGBM (AUC = {roc_auc:.3f})')

    # Neural Network
    if 'ann' in models_dict:
        X_test_scaled = models_dict['ann']['scaler'].transform(X_test)
        y_proba = models_dict['ann']['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Neural Network (AUC = {roc_auc:.3f})')

    # SVM
    if 'svm' in models_dict:
        X_test_scaled = models_dict['svm']['scaler'].transform(X_test)
        y_proba = models_dict['svm']['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.3f})')

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Classification Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('A2/roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    print("\nROC curves saved as 'roc_curves_all_models.png'")
    plt.close()


def compare_all_models(models_dict):
    """
    Create a comparison table of all model accuracies.

    Parameters:
    -----------
    models_dict : dict
        Dictionary containing all trained models with their accuracies
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    comparison_data = []

    model_names = {
        'decision_tree': 'Decision Tree',
        'naive_bayes': 'Naive Bayes',
        'bagging': 'Bagging',
        'boosting': 'AdaBoost',
        'random_forest': 'Random Forest',
        'pruned_dt': 'Pruned Decision Tree',
        'lightgbm': 'LightGBM',
        'ann': 'Neural Network (ANN)',
        'svm': 'Support Vector Machine'
    }

    for key, name in model_names.items():
        if key in models_dict:
            comparison_data.append({
                'Model': name,
                'Accuracy': models_dict[key]['accuracy']
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

    print("\n")
    print(comparison_df.to_string(index=False))

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(comparison_df)), comparison_df['Accuracy'])
    plt.yticks(range(len(comparison_df)), comparison_df['Model'])
    plt.xlabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xlim(0, 1.0)

    # Add accuracy values on bars
    for i, v in enumerate(comparison_df['Accuracy']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('A2/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison plot saved as 'model_comparison.png'")
    plt.close()

    return comparison_df


def main():
    """
    Main function to run the complete analysis pipeline.
    """
    print("\n" + "="*80)
    print("FIT3152 ASSIGNMENT 2 - PHISHING WEBSITE DETECTION")
    print("Student ID: 42857193")
    print("="*80)

    # Load data
    data = load_data('PhishingData.csv')

    # Question 1: Dataset dimensions
    question1_dimensions(data)

    # Question 2: Sample data
    PD = question2_sample_data(data)

    # Question 3: Check missing values
    question3_missing_values(PD)

    # Question 4: Handle missing values
    PD_clean = question4_handle_missing(PD)

    # Question 5: Train-test split
    X_train, X_test, y_train, y_test = question5_train_test_split(PD_clean)

    # Dictionary to store all models
    all_models = {}

    # Question 6: Decision Tree
    dt_model, dt_acc, dt_pred = question6_decision_tree(X_train, X_test, y_train, y_test)
    all_models['decision_tree'] = {'model': dt_model, 'accuracy': dt_acc, 'predictions': dt_pred}

    # Question 7: Naive Bayes
    nb_model, nb_acc, nb_pred = question7_naive_bayes(X_train, X_test, y_train, y_test)
    all_models['naive_bayes'] = {'model': nb_model, 'accuracy': nb_acc, 'predictions': nb_pred}

    # Question 8: Bagging
    bag_model, bag_acc, bag_pred = question8_bagging(X_train, X_test, y_train, y_test)
    all_models['bagging'] = {'model': bag_model, 'accuracy': bag_acc, 'predictions': bag_pred}

    # Question 9: Boosting
    boost_model, boost_acc, boost_pred = question9_boosting(X_train, X_test, y_train, y_test)
    all_models['boosting'] = {'model': boost_model, 'accuracy': boost_acc, 'predictions': boost_pred}

    # Question 10: Random Forest
    rf_model, rf_acc, rf_pred = question10_random_forest(X_train, X_test, y_train, y_test)
    all_models['random_forest'] = {'model': rf_model, 'accuracy': rf_acc, 'predictions': rf_pred}

    # Question 11: Advanced models
    advanced_results = question11_advanced_models(X_train, X_test, y_train, y_test)
    all_models.update(advanced_results)

    # Question 12: Neural Network and SVM
    nn_svm_results = question12_neural_network_svm(X_train, X_test, y_train, y_test)
    all_models.update(nn_svm_results)

    # Plot ROC curves
    plot_roc_curves(all_models, X_test, y_test)

    # Compare all models
    comparison_df = compare_all_models(all_models)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - decision_tree_confusion.png")
    print("  - random_forest_importance.png")
    print("  - roc_curves_all_models.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
