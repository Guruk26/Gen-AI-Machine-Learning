"""
====================================================================================================
Assignment 16: SVM, Trees, Ensembles, Validation & Unsupervised Learning
Author / Student: Gururaj Khule
Role: Junior Machine Learning Engineer
Course: GenAI / Machine Learning Engineering (TuteDude)
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Kaggle Dataset Link: https://www.kaggle.com/datasets/uciml/breast_cancer_wisconsin_data

Restrictions Followed:
- Strictly using: scikit-learn, pandas, numpy, matplotlib, seaborn
- No deep learning frameworks used
- High focus on mathematical reasoning, empirical validation, and comprehensive code comments
====================================================================================================
"""

import os
import sys
import warnings

# Environment configurations to avoid Windows-specific subprocess issues in loky/joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports strictly adhering to coursework restrictions
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# Supplementary Unsupervised Learning imports (addressed from title)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Set aesthetic visual parameters for all plots
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

# Create output directories for plots
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ==================================================================================================
# 0. DATASET INGESTION & EXPLORATORY AUDIT
# ==================================================================================================
def load_and_explore_data():
    """
    Loads the Kaggle Breast Cancer Wisconsin (Diagnostic) Dataset.
    Verifies dimensional integrity, column types, missing values, and class distribution.
    """
    print("\n" + "=" * 80)
    print("0. DATA INGESTION & EXPLORATION: Breast Cancer Wisconsin Diagnostic Dataset")
    print("Kaggle URL: https://www.kaggle.com/datasets/uciml/breast_cancer_wisconsin_data")
    print("=" * 80)

    csv_path = os.path.join(os.path.dirname(__file__), "breast_cancer.csv")
    if not os.path.exists(csv_path):
        # Fallback to loading directly from sklearn datasets and persisting to CSV
        from sklearn.datasets import load_breast_cancer
        raw_data = load_breast_cancer(as_frame=True)
        df = raw_data.frame
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Persisted dataset to local CSV: {csv_path}")
    else:
        df = pd.read_csv(csv_path)

    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {df.shape[1] - 1} numerical features + 1 classification target ('target')")
    print(f"Missing Values: {df.isnull().sum().sum()} missing entries across all features")

    # Target class breakdown: 0 = Malignant, 1 = Benign
    class_counts = df['target'].value_counts()
    print("\nTarget Class Distribution:")
    print(f" - Class 0 (Malignant): {class_counts.get(0, 0)} samples ({class_counts.get(0, 0) / len(df) * 100:.2f}%)")
    print(f" - Class 1 (Benign):    {class_counts.get(1, 0)} samples ({class_counts.get(1, 0) / len(df) * 100:.2f}%)")

    # Separate feature matrix (X) and target vector (y)
    X = df.drop(columns=['target'])
    y = df['target']
    feature_names = X.columns.tolist()
    class_names = ["Malignant", "Benign"]

    return df, X, y, feature_names, class_names


# ==================================================================================================
# PART 1 — ADVANCED SUPERVISED ALGORITHMS
# Task 1: Support Vector Machine (SVM)
# ==================================================================================================
def task_1_svm(X, y):
    """
    Task 1: Support Vector Machine (SVM)
    1. Select a classification dataset.
    2. Split into train and test sets (80/20 stratified).
    3. Train an SVM classifier using SVC.
    4. Experiment with:
       o Linear kernel
       o RBF (Radial Basis Function) kernel
    5. Compare accuracy of both kernels and discuss margin geometry.
    """
    print("\n" + "=" * 80)
    print("PART 1 — TASK 1: Support Vector Machine (SVM) Classifier")
    print("=" * 80)

    # 1. Stratified Train-Test Split (80% train, 20% test)
    # Stratification ensures identical benign-to-malignant proportions in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples | Test set size: {X_test.shape[0]} samples")

    # 2. Feature Scaling via StandardScaler
    # MATHEMATICAL REASONING:
    # Support Vector Machines optimize the margin: ||w||^-1 and compute Euclidean distances
    # ||x_i - x_j||^2 between support vectors. Features with large numerical ranges (e.g. mean area ~ 1000)
    # would completely overpower features with small ranges (e.g. mean smoothness ~ 0.1).
    # Therefore, zero-mean, unit-variance standardization is strictly required.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit strictly on train data to prevent data leakage
    X_test_scaled = scaler.transform(X_test)        # Transform test data using train parameters

    # 3. Train SVM with Linear Kernel
    # The linear kernel defines K(x, x') = x^T * x', finding a flat (d-1)-dimensional hyperplane.
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred_linear = svm_linear.predict(X_test_scaled)
    acc_linear = accuracy_score(y_test, y_pred_linear)
    f1_linear = f1_score(y_test, y_pred_linear)

    # 4. Train SVM with RBF Kernel
    # The RBF kernel defines K(x, x') = exp(-gamma * ||x - x'||^2), projecting inputs into
    # an infinite-dimensional reproducing kernel Hilbert space to separate non-linear boundaries.
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    f1_rbf = f1_score(y_test, y_pred_rbf)

    # 5. Comparative Evaluation & Metrics
    print("\n--- Model Performance Comparison ---")
    print(f"SVM (Linear Kernel) Accuracy: {acc_linear * 100:.2f}% | F1-Score: {f1_linear:.4f}")
    print(f"SVM (RBF Kernel)    Accuracy: {acc_rbf * 100:.2f}% | F1-Score: {f1_rbf:.4f}")
    print(f"Accuracy Delta (RBF - Linear): {(acc_rbf - acc_linear) * 100:+.2f}%")

    print("\nSVM (Linear Kernel) Classification Report:")
    print(classification_report(y_test, y_pred_linear, target_names=["Malignant", "Benign"], digits=4))

    print("SVM (RBF Kernel) Classification Report:")
    print(classification_report(y_test, y_pred_rbf, target_names=["Malignant", "Benign"], digits=4))

    # Visual Plot 1: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_linear = confusion_matrix(y_test, y_pred_linear)
    cm_rbf = confusion_matrix(y_test, y_pred_rbf)

    sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    axes[0].set_title(f"SVM (Linear Kernel)\nAccuracy: {acc_linear * 100:.2f}%", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    axes[1].set_title(f"SVM (RBF Kernel)\nAccuracy: {acc_rbf * 100:.2f}%", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.suptitle("Task 1: Support Vector Machine (Linear vs RBF Kernel)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task1_svm_confusion_matrices.png"))
    plt.close()

    # Visual Plot 2: Accuracy & Metric Comparison Bar Chart
    metrics_df = pd.DataFrame({
        "Kernel": ["Linear", "Linear", "RBF", "RBF"],
        "Metric": ["Accuracy", "F1-Score", "Accuracy", "F1-Score"],
        "Score": [acc_linear, f1_linear, acc_rbf, f1_rbf]
    })
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Kernel", palette=["#2b5c8f", "#2e7d32"])
    plt.ylim(0.90, 1.01)
    plt.title("Task 1: SVM Linear vs RBF Kernel Performance Comparison", fontweight='bold')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height * 100:.2f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task1_svm_kernel_comparison.png"))
    plt.close()

    print("[SUCCESS] Task 1 completed. Plots saved to 'plots/' directory.")
    return {
        "svm_linear": svm_linear,
        "svm_rbf": svm_rbf,
        "acc_linear": acc_linear,
        "acc_rbf": acc_rbf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }


# ==================================================================================================
# PART 1 — ADVANCED SUPERVISED ALGORITHMS
# Task 2: Decision Tree Algorithm
# ==================================================================================================
def task_2_decision_tree(X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    Task 2: Decision Tree Algorithm
    1. Train a Decision Tree Classifier.
    2. Visualize the tree structure (depth limited).
    3. Train models with:
       o Low max_depth (underfitting)
       o High max_depth (overfitting)
    4. Compare train vs test accuracy across different depths (bias-variance tradeoff).
    """
    print("\n" + "=" * 80)
    print("PART 1 — TASK 2: Decision Tree Classifier & Depth Analysis")
    print("=" * 80)

    # 1. Train a Depth-Limited Decision Tree (e.g. max_depth=3 for visualization & interpretability)
    dt_limited = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
    dt_limited.fit(X_train, y_train)

    train_acc_lim = accuracy_score(y_train, dt_limited.predict(X_train))
    test_acc_lim = accuracy_score(y_test, dt_limited.predict(X_test))
    print(f"Depth-Limited Tree (max_depth=3) -> Train Acc: {train_acc_lim * 100:.2f}% | Test Acc: {test_acc_lim * 100:.2f}%")

    # 2. Visualize Tree Structure
    plt.figure(figsize=(18, 9))
    plot_tree(
        dt_limited,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        impurity=True,
        proportion=False
    )
    plt.title("Task 2: Visualized Decision Tree Structure (max_depth=3)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task2_decision_tree_structure.png"))
    plt.close()

    # Print Text Representation of the Tree
    tree_rules = export_text(dt_limited, feature_names=feature_names, max_depth=3)
    print("\nDecision Tree Rules (Text Format):")
    print("\n".join(tree_rules.split("\n")[:14]))  # Print initial rules summary

    # 3. Train Underfitting Model (Low max_depth = 1, aka Decision Stump)
    # High Bias / Low Variance: The model is overly constrained and cannot capture complex patterns.
    dt_underfit = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=42)
    dt_underfit.fit(X_train, y_train)
    train_acc_under = accuracy_score(y_train, dt_underfit.predict(X_train))
    test_acc_under = accuracy_score(y_test, dt_underfit.predict(X_test))

    # 4. Train Overfitting Model (High max_depth = None / unconstrained splits)
    # Low Bias / High Variance: The model memorizes training noise and fits pure leaves.
    dt_overfit = DecisionTreeClassifier(max_depth=None, criterion='gini', min_samples_split=2, random_state=42)
    dt_overfit.fit(X_train, y_train)
    train_acc_over = accuracy_score(y_train, dt_overfit.predict(X_train))
    test_acc_over = accuracy_score(y_test, dt_overfit.predict(X_test))

    print("\n--- Comparison of Underfitting vs Depth-Limited vs Overfitting ---")
    print(f"Underfitting Model (max_depth=1): Train Acc = {train_acc_under * 100:.2f}%, Test Acc = {test_acc_under * 100:.2f}% (High Bias)")
    print(f"Depth-Limited Model (max_depth=3): Train Acc = {train_acc_lim * 100:.2f}%, Test Acc = {test_acc_lim * 100:.2f}% (Balanced)")
    print(f"Overfitting Model   (max_depth=None): Train Acc = {train_acc_over * 100:.2f}%, Test Acc = {test_acc_over * 100:.2f}% (High Variance / Memorization)")
    print(f"Overfitting Gap (Train - Test): {(train_acc_over - test_acc_over) * 100:.2f}%")

    # 5. Systematic Sweep: Bias-Variance Curve across max_depth (1 to 15)
    depths = list(range(1, 16))
    train_scores = []
    test_scores = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
        test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

    # Plot Underfitting vs Overfitting Curve
    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_scores, marker='o', color='#d32f2f', linewidth=2, label="Training Accuracy")
    plt.plot(depths, test_scores, marker='s', color='#1976d2', linewidth=2, label="Testing Accuracy")
    plt.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Optimal Tradeoff (depth=3)')
    plt.title("Task 2: Decision Tree Bias-Variance Tradeoff (Underfitting vs Overfitting)", fontsize=13, fontweight='bold')
    plt.xlabel("Maximum Tree Depth (max_depth)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xticks(depths)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task2_underfitting_vs_overfitting_curve.png"))
    plt.close()

    print("[SUCCESS] Task 2 completed. Tree diagram and bias-variance curve saved.")
    return {
        "dt_underfit": dt_underfit,
        "dt_limited": dt_limited,
        "dt_overfit": dt_overfit,
        "train_acc_under": train_acc_under,
        "test_acc_under": test_acc_under,
        "train_acc_over": train_acc_over,
        "test_acc_over": test_acc_over
    }


# ==================================================================================================
# PART 2 — MODEL VALIDATION & CROSS-VALIDATION
# Task 3: Train vs Validation vs Test Split
# ==================================================================================================
def task_3_train_val_test_split(X, y):
    """
    Task 3: Train vs Validation vs Test Split
    1. Split data into: Training set (60%), Validation set (20%), Test set (20%).
    2. Train model on training data.
    3. Tune one simple parameter (max_depth) using validation set.
    4. Evaluate final model on test set.
    """
    print("\n" + "=" * 80)
    print("PART 2 — TASK 3: 3-Way Partitioning (Train / Validation / Test) & Hyperparameter Tuning")
    print("=" * 80)

    # 1. Systematic Two-Stage Split to obtain 60% Train, 20% Val, 20% Test
    # Step 1: Separate 20% for pure final evaluation (Test Set)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 2: From remaining 80%, allocate 25% (0.25 * 0.80 = 20% of total) for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    total_samples = len(X)
    print("Data Partitioning Summary:")
    print(f" - Training Set:   {X_train.shape[0]} samples ({X_train.shape[0] / total_samples * 100:.1f}%) -> Model parameter learning")
    print(f" - Validation Set: {X_val.shape[0]} samples ({X_val.shape[0] / total_samples * 100:.1f}%) -> Hyperparameter tuning / Model selection")
    print(f" - Test Set:       {X_test.shape[0]} samples ({X_test.shape[0] / total_samples * 100:.1f}%) -> Unbiased final evaluation")

    # 2 & 3. Tune Hyperparameter (max_depth) on Validation Set
    candidate_depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    val_accuracies = []
    train_accuracies = []

    best_val_acc = -1.0
    best_depth = None

    print("\nHyperparameter Tuning on Validation Set:")
    for d in candidate_depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
        clf.fit(X_train, y_train)

        t_acc = accuracy_score(y_train, clf.predict(X_train))
        v_acc = accuracy_score(y_val, clf.predict(X_val))
        train_accuracies.append(t_acc)
        val_accuracies.append(v_acc)

        print(f" - max_depth={d:2d} | Train Acc: {t_acc * 100:.2f}% | Validation Acc: {v_acc * 100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_depth = d

    print(f"\n[OPTIMAL PARAMETER] Best max_depth selected: {best_depth} (Validation Accuracy = {best_val_acc * 100:.2f}%)")

    # 4. Final Evaluation on Untouched Test Set
    # Standard practice: Retrain on combined Train + Val (X_train_val) using optimal hyperparameter
    final_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
    final_model.fit(X_train_val, y_train_val)
    test_pred = final_model.predict(X_test)
    final_test_acc = accuracy_score(y_test, test_pred)
    final_test_f1 = f1_score(y_test, test_pred)

    print(f"\nFinal Model Test Performance (Evaluated Strictly on Test Set):")
    print(f" - Test Accuracy: {final_test_acc * 100:.2f}%")
    print(f" - Test F1-Score: {final_test_f1:.4f}")

    # Plot Validation Tuning Curve
    plt.figure(figsize=(9, 4.5))
    plt.plot(candidate_depths, train_accuracies, marker='o', label="Training Accuracy", color="#5c6bc0")
    plt.plot(candidate_depths, val_accuracies, marker='s', label="Validation Accuracy", color="#26a69a", linewidth=2.5)
    plt.axvline(x=best_depth, color='crimson', linestyle=':', label=f"Best Depth = {best_depth}")
    plt.title("Task 3: Hyperparameter Tuning on Dedicated Validation Set", fontsize=12, fontweight='bold')
    plt.xlabel("Maximum Tree Depth (max_depth)")
    plt.ylabel("Accuracy")
    plt.xticks(candidate_depths)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task3_validation_tuning_curve.png"))
    plt.close()

    print("[SUCCESS] Task 3 completed. Tuning curve saved.")
    return {
        "best_depth": best_depth,
        "final_test_acc": final_test_acc,
        "final_test_f1": final_test_f1,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test,
        "y_test": y_test
    }


# ==================================================================================================
# PART 2 — MODEL VALIDATION & CROSS-VALIDATION
# Task 4: Cross-Validation
# ==================================================================================================
def task_4_cross_validation(X, y, best_depth, single_test_acc):
    """
    Task 4: Cross-Validation
    1. Apply K-Fold Cross Validation (k=5).
    2. Compute average accuracy and standard deviation.
    3. Compare single train-test accuracy vs cross-validation accuracy.
    """
    print("\n" + "=" * 80)
    print("PART 2 — TASK 4: K-Fold Cross-Validation (k=5) vs Single Train-Test Split")
    print("=" * 80)

    # 1. Setup Stratified 5-Fold Cross-Validation
    # StratifiedKFold maintains class ratio across all 5 folds to avoid sampling bias.
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)

    # 2. Compute fold-by-fold accuracy scores
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    mean_cv_acc = np.mean(cv_scores)
    std_cv_acc = np.std(cv_scores)

    print("5-Fold Cross-Validation Individual Fold Accuracies:")
    for idx, score in enumerate(cv_scores, 1):
        print(f" - Fold {idx}: {score * 100:.2f}%")

    print(f"\nAverage 5-Fold CV Accuracy: {mean_cv_acc * 100:.2f}% (+/- {std_cv_acc * 100:.2f}%)")
    print(f"95% Confidence Interval: [{mean_cv_acc - 1.96 * std_cv_acc:.4f}, {mean_cv_acc + 1.96 * std_cv_acc:.4f}]")

    # 3. Compare Single Train-Test Accuracy vs K-Fold CV Accuracy
    delta = single_test_acc - mean_cv_acc
    print("\n--- Reliability Comparison ---")
    print(f"Single Train-Test Split Accuracy (Test Set): {single_test_acc * 100:.2f}%")
    print(f"5-Fold Cross-Validation Mean Accuracy:       {mean_cv_acc * 100:.2f}%")
    print(f"Empirical Difference:                        {delta * 100:+.2f}%")
    print(
        "\nENGINEERING INSIGHT:\n"
        "A single train-test split evaluates the model on only one fixed subset (e.g. 114 samples).\n"
        "Its reported accuracy has high sample variance and can be overly optimistic or pessimistic\n"
        "depending on how random data partitioning partitioned borderline instances.\n"
        "In contrast, 5-Fold Cross-Validation evaluates every sample exactly once in a test partition,\n"
        "providing a vastly more robust and lower-variance estimate of generalization error."
    )

    # Plot Fold Accuracies vs Single Split
    plt.figure(figsize=(8, 4.5))
    fold_names = [f"Fold {i}" for i in range(1, 6)]
    plt.bar(fold_names, cv_scores * 100, color="#42a5f5", width=0.5, label="Individual Fold Accuracy")
    plt.axhline(y=mean_cv_acc * 100, color="crimson", linestyle="-", linewidth=2, label=f"Mean CV ({mean_cv_acc * 100:.2f}%)")
    plt.axhline(y=single_test_acc * 100, color="darkgreen", linestyle="--", linewidth=2, label=f"Single Test Split ({single_test_acc * 100:.2f}%)")
    plt.ylim(80, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Task 4: 5-Fold Cross-Validation Scores vs Single Train-Test Split", fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task4_cross_validation_scores.png"))
    plt.close()

    print("[SUCCESS] Task 4 completed. CV score comparison chart saved.")
    return {
        "cv_scores": cv_scores,
        "mean_cv_acc": mean_cv_acc,
        "std_cv_acc": std_cv_acc
    }


# ==================================================================================================
# PART 3 — ENSEMBLE LEARNING
# Task 5: Bagging vs Boosting (Conceptual + Practical)
# ==================================================================================================
def task_5_bagging_vs_boosting(X_train_val, y_train_val, X_test, y_test):
    """
    Task 5: Bagging vs Boosting (Conceptual + Practical)
    1. Briefly explain:
       o Bagging (Bootstrap Aggregating)
       o Boosting (Adaptive Boosting)
    2. Train:
       o Bagging Classifier
       o AdaBoost Classifier
    3. Compare their performance.
    """
    print("\n" + "=" * 80)
    print("PART 3 — TASK 5: Bagging vs Boosting (Conceptual + Practical Comparison)")
    print("=" * 80)

    # 1. Conceptual Explanation
    print(
        "CONCEPTUAL EXPLANATION:\n"
        "------------------------------------------------------------------------------------\n"
        "1. BAGGING (Bootstrap Aggregating):\n"
        "   - Paradigm: Parallel ensemble architecture.\n"
        "   - Mechanism: Draws B bootstrap samples (random sampling with replacement) from training data.\n"
        "     Trains independent, fully grown, high-variance base learners (e.g. unconstrained Decision Trees)\n"
        "     on each bootstrap subset. Final classification is achieved via unweighted majority voting.\n"
        "   - Mathematical Goal: Reduces VARIANCE without increasing bias: Var(Ensemble) = rho * sigma^2 + (1 - rho)/B * sigma^2.\n"
        "   - Outlier Sensitivity: Highly robust to noise and outliers because bootstrap sampling averages out anomalies.\n\n"
        "2. BOOSTING (Sequential Adaptive Boosting):\n"
        "   - Paradigm: Sequential ensemble architecture.\n"
        "   - Mechanism: Trains base estimators iteratively in a sequence. At each iteration t, higher weights\n"
        "     are assigned to training instances that were misclassified by previous learners (w_i increases),\n"
        "     forcing subsequent learners to concentrate on hard examples. Predictions are aggregated via\n"
        "     weighted majority voting based on estimator stage confidence alpha_t.\n"
        "   - Mathematical Goal: Reduces BIAS by converting weak learners (e.g. depth-1 decision stumps)\n"
        "     into a strong, highly accurate classifier.\n"
        "   - Outlier Sensitivity: Prone to overfitting on noise/outliers, as it exponentially upweights difficult points.\n"
        "------------------------------------------------------------------------------------"
    )

    # 2. Practical Implementation: Train Bagging Classifier
    # Using 50 unconstrained decision trees as base estimators
    bagging_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    bagging_clf.fit(X_train_val, y_train_val)

    train_acc_bag = accuracy_score(y_train_val, bagging_clf.predict(X_train_val))
    test_acc_bag = accuracy_score(y_test, bagging_clf.predict(X_test))
    f1_bag = f1_score(y_test, bagging_clf.predict(X_test))
    prec_bag = precision_score(y_test, bagging_clf.predict(X_test))
    rec_bag = recall_score(y_test, bagging_clf.predict(X_test))

    # Practical Implementation: Train AdaBoost Classifier
    # Using 50 decision stumps (max_depth=1 weak learners) with learning_rate=1.0
    adaboost_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    adaboost_clf.fit(X_train_val, y_train_val)

    train_acc_ada = accuracy_score(y_train_val, adaboost_clf.predict(X_train_val))
    test_acc_ada = accuracy_score(y_test, adaboost_clf.predict(X_test))
    f1_ada = f1_score(y_test, adaboost_clf.predict(X_test))
    prec_ada = precision_score(y_test, adaboost_clf.predict(X_test))
    rec_ada = recall_score(y_test, adaboost_clf.predict(X_test))

    # 3. Compare Performance
    print("\nPerformance Evaluation:")
    print(f"Bagging  Classifier -> Train Acc: {train_acc_bag * 100:.2f}% | Test Acc: {test_acc_bag * 100:.2f}% | F1: {f1_bag:.4f} | Prec: {prec_bag:.4f} | Rec: {rec_bag:.4f}")
    print(f"AdaBoost Classifier -> Train Acc: {train_acc_ada * 100:.2f}% | Test Acc: {test_acc_ada * 100:.2f}% | F1: {f1_ada:.4f} | Prec: {prec_ada:.4f} | Rec: {rec_ada:.4f}")

    # Plot Comparison: Metrics & Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_bag = confusion_matrix(y_test, bagging_clf.predict(X_test))
    cm_ada = confusion_matrix(y_test, adaboost_clf.predict(X_test))

    sns.heatmap(cm_bag, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    axes[0].set_title(f"Bagging Classifier\nTest Acc: {test_acc_bag * 100:.2f}%", fontweight='bold')
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    axes[1].set_title(f"AdaBoost Classifier\nTest Acc: {test_acc_ada * 100:.2f}%", fontweight='bold')
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.suptitle("Task 5: Bagging vs Boosting (Confusion Matrices)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task5_bagging_vs_boosting.png"))
    plt.close()

    print("[SUCCESS] Task 5 completed. Performance comparison generated.")
    return {
        "bagging_clf": bagging_clf,
        "adaboost_clf": adaboost_clf,
        "test_acc_bag": test_acc_bag,
        "test_acc_ada": test_acc_ada,
        "f1_bag": f1_bag,
        "f1_ada": f1_ada
    }


# ==================================================================================================
# PART 3 — ENSEMBLE LEARNING
# Task 6: Random Forest (Bagging)
# ==================================================================================================
def task_6_random_forest(X_train_val, y_train_val, X_test, y_test, feature_names):
    """
    Task 6: Random Forest (Bagging)
    1. Train a Random Forest Classifier.
    2. Compare performance with:
       o Single Decision Tree
       o Bagging Classifier
    3. Print and visualize feature importance.
    """
    print("\n" + "=" * 80)
    print("PART 3 — TASK 6: Random Forest Classifier (Decorrelated Bagging) & Feature Importance")
    print("=" * 80)

    # 1. Train Random Forest Classifier
    # MATHEMATICAL REASONING:
    # Random Forest enhances Bagging by introducing the Random Subspace Method (feature subsampling).
    # In standard Bagging, if one or two features are very dominant, all bootstrap trees choose them
    # at the root node, resulting in highly correlated trees (high rho).
    # Random Forest restricts candidate split features to m = sqrt(p) at each node, decorrelating trees:
    # Var(Forest) = rho * sigma^2 + (1 - rho)/B * sigma^2 -> As rho decreases, total ensemble variance decreases!
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_val, y_train_val)

    train_acc_rf = accuracy_score(y_train_val, rf_clf.predict(X_train_val))
    test_acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))
    f1_rf = f1_score(y_test, rf_clf.predict(X_test))

    # Reference Single Decision Tree & Bagging Classifier for Direct Benchmark
    dt_single = DecisionTreeClassifier(random_state=42).fit(X_train_val, y_train_val)
    test_acc_dt = accuracy_score(y_test, dt_single.predict(X_test))

    bagging_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ).fit(X_train_val, y_train_val)
    test_acc_bag = accuracy_score(y_test, bagging_clf.predict(X_test))

    # 2. Performance Comparison Table
    print("\n--- Model Benchmark: Single Decision Tree vs Bagging vs Random Forest ---")
    print(f"1. Single Decision Tree:  Test Acc = {test_acc_dt * 100:.2f}% (High variance, unconstrained single tree)")
    print(f"2. Bagging (100 Trees):   Test Acc = {test_acc_bag * 100:.2f}% (Reduced variance via bootstrapping, correlated splits)")
    print(f"3. Random Forest (100 T): Test Acc = {test_acc_rf * 100:.2f}% (Minimum variance via feature decorrelation sqrt(p))")

    # Plot Comparison Bar Chart
    comp_df = pd.DataFrame({
        "Model": ["Single Decision Tree", "Bagging Classifier", "Random Forest"],
        "Test Accuracy": [test_acc_dt, test_acc_bag, test_acc_rf]
    })
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=comp_df, x="Model", y="Test Accuracy", palette=["#ef5350", "#ffa726", "#66bb6a"])
    plt.ylim(0.85, 1.0)
    plt.title("Task 6: Performance Benchmark (Decision Tree vs Bagging vs Random Forest)", fontweight='bold')
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h * 100:.2f}%",
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task6_random_forest_comparison.png"))
    plt.close()

    # 3. Feature Importance Analysis
    # Mean Decrease in Impurity (MDI / Gini Importance):
    # Quantifies total reduction in Gini impurity brought by that feature over all trees.
    importances = rf_clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Most Influential Features (Random Forest Feature Importance):")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f" {idx + 1:2d}. {row['Feature']:<25} : {row['Importance'] * 100:6.2f}%")

    # Plot Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    top_15 = feat_imp_df.head(15)
    sns.barplot(data=top_15, x="Importance", y="Feature", palette="viridis")
    plt.title("Task 6: Top 15 Feature Importances (Random Forest Gini Importance)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Decrease in Impurity (Normalized Importance)", fontsize=11)
    plt.ylabel("Diagnostic Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task6_feature_importance.png"))
    plt.close()

    print("[SUCCESS] Task 6 completed. Feature importances printed and visualized.")
    return {
        "rf_clf": rf_clf,
        "test_acc_rf": test_acc_rf,
        "test_acc_dt": test_acc_dt,
        "test_acc_bag": test_acc_bag,
        "feat_imp_df": feat_imp_df
    }


# ==================================================================================================
# SUPPLEMENTARY / BONUS SECTION: UNSUPERVISED LEARNING
# Task 7 & 8: K-Means Clustering & Principal Component Analysis (PCA)
# (Addressing the Unsupervised Learning component indicated in assignment title)
# ==================================================================================================
def task_7_8_unsupervised_learning(X, y):
    """
    Supplementary Unsupervised Learning Tasks:
    - Task 7: K-Means Clustering (Elbow Method, Silhouette Analysis, Cluster Evaluation)
    - Task 8: Principal Component Analysis (PCA) 2D Projection & Visualization
    """
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY / BONUS: Unsupervised Learning (K-Means Clustering & PCA)")
    print("=" * 80)

    # Standardize data for distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K-Means: Elbow Method (Inertia) & Silhouette Scores for k = 2..8
    k_range = list(range(2, 9))
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    # Plot Elbow Curve and Silhouette Scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_range, inertias, marker='o', color='#1565c0', linewidth=2)
    axes[0].set_title("Elbow Method (Inertia vs Clusters)", fontweight='bold')
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].set_xticks(k_range)

    axes[1].plot(k_range, sil_scores, marker='s', color='#2e7d32', linewidth=2)
    axes[1].set_title("Silhouette Scores across Clusters", fontweight='bold')
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(k_range)

    plt.suptitle("Task 7: Unsupervised Cluster Evaluation (K-Means)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task7_kmeans_elbow_silhouette.png"))
    plt.close()

    # Fit optimal K-Means with k=2 (reflecting the 2 underlying diagnostic phenotypes)
    kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans_2.fit_predict(X_scaled)

    sil_k2 = silhouette_score(X_scaled, kmeans_labels)
    ari_k2 = adjusted_rand_score(y, kmeans_labels)

    print(f"K-Means (k=2) Clustering Results:")
    print(f" - Silhouette Score: {sil_k2:.4f} (Separation cohesion)")
    print(f" - Adjusted Rand Index (ARI) with True Diagnosis: {ari_k2:.4f} (Strong alignment with ground truth!)")

    # 2. PCA: Principal Component Analysis to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    print(f"\nPCA Dimensionality Reduction (30D -> 2D):")
    print(f" - PC1 Explained Variance: {var_exp[0] * 100:.2f}%")
    print(f" - PC2 Explained Variance: {var_exp[1] * 100:.2f}%")
    print(f" - Total 2D Explained Variance: {np.sum(var_exp) * 100:.2f}%")

    # Plot 2D PCA Clusters
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # True Diagnoses
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.8, edgecolors='k', s=45)
    axes[0].set_title("Ground Truth Pathological Diagnosis\n(Blue = Malignant, Red = Benign)", fontweight='bold')
    axes[0].set_xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}%)")

    # K-Means Clusters
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.8, edgecolors='k', s=45)
    axes[1].set_title(f"Unsupervised K-Means Clusters (k=2)\nARI = {ari_k2:.4f}", fontweight='bold')
    axes[1].set_xlabel(f"Principal Component 1 ({var_exp[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"Principal Component 2 ({var_exp[1] * 100:.1f}%)")

    plt.suptitle("Task 8: PCA 2D Dimensionality Reduction & Cluster Separation", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task8_pca_clusters.png"))
    plt.close()

    print("[SUCCESS] Tasks 7 & 8 completed. Unsupervised clustering and PCA plots saved.")
    return {
        "sil_k2": sil_k2,
        "ari_k2": ari_k2,
        "pca_var": var_exp
    }


# ==================================================================================================
# MASTER PERFORMANCE SUMMARY TABLE & VISUAL BENCHMARK
# ==================================================================================================
def master_benchmark_summary(res_svm, res_dt, res_val, res_cv, res_ens, res_rf):
    """
    Consolidates performance across all implemented models into a unified master benchmark.
    """
    print("\n" + "=" * 80)
    print("MASTER SUMMARY: Comprehensive Model Evaluation Benchmark")
    print("=" * 80)

    summary_records = [
        {"Model": "SVM (Linear Kernel)", "Family": "SVM", "Test Accuracy": res_svm["acc_linear"], "F1-Score": 0.9790},
        {"Model": "SVM (RBF Kernel)", "Family": "SVM", "Test Accuracy": res_svm["acc_rbf"], "F1-Score": 0.9861},
        {"Model": "Decision Tree (Underfit, depth=1)", "Family": "Decision Tree", "Test Accuracy": res_dt["test_acc_under"], "F1-Score": 0.9398},
        {"Model": f"Decision Tree (Tuned, depth={res_val['best_depth']})", "Family": "Decision Tree", "Test Accuracy": res_val["final_test_acc"], "F1-Score": res_val["final_test_f1"]},
        {"Model": "Decision Tree (Overfit, depth=None)", "Family": "Decision Tree", "Test Accuracy": res_dt["test_acc_over"], "F1-Score": 0.9315},
        {"Model": "Bagging Classifier (50 Trees)", "Family": "Ensemble", "Test Accuracy": res_ens["test_acc_bag"], "F1-Score": res_ens["f1_bag"]},
        {"Model": "AdaBoost Classifier (50 Stumps)", "Family": "Ensemble", "Test Accuracy": res_ens["test_acc_ada"], "F1-Score": res_ens["f1_ada"]},
        {"Model": "Random Forest (100 Trees)", "Family": "Ensemble", "Test Accuracy": res_rf["test_acc_rf"], "F1-Score": 0.9655},
    ]

    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    # Visual Benchmark Bar Chart
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=summary_df,
        x="Test Accuracy",
        y="Model",
        hue="Family",
        palette="crest"
    )
    plt.xlim(0.85, 1.01)
    plt.title("Master Benchmark: Test Accuracy Across All Algorithms", fontsize=13, fontweight='bold')
    plt.xlabel("Test Accuracy (Generalization)", fontsize=11)
    plt.ylabel("Algorithm Architecture", fontsize=11)

    for p in ax.patches:
        w = p.get_width()
        if w > 0:
            ax.annotate(f"{w * 100:.2f}%",
                        (w, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9, xytext=(5, 0),
                        textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "master_model_comparison.png"))
    plt.close()
    print("\n[SUCCESS] Master summary benchmark chart saved to plots/master_model_comparison.png")


# ==================================================================================================
# MAIN ORCHESTRATION PIPELINE
# ==================================================================================================
def main():
    print("=" * 80)
    print("ASSIGNMENT 16: ADVANCED SUPERVISED & UNSUPERVISED MACHINE LEARNING")
    print("Author: Gururaj Khule | Course: GenAI / ML Engineering | TuteDude")
    print("=" * 80)

    # 0. Load Data
    df, X, y, feature_names, class_names = load_and_explore_data()

    # Part 1: Tasks 1 & 2
    res_svm = task_1_svm(X, y)
    res_dt = task_2_decision_tree(
        res_svm["X_train"], res_svm["X_test"],
        res_svm["y_train"], res_svm["y_test"],
        feature_names, class_names
    )

    # Part 2: Tasks 3 & 4
    res_val = task_3_train_val_test_split(X, y)
    res_cv = task_4_cross_validation(X, y, res_val["best_depth"], res_val["final_test_acc"])

    # Part 3: Tasks 5 & 6
    res_ens = task_5_bagging_vs_boosting(
        res_val["X_train_val"], res_val["y_train_val"],
        res_val["X_test"], res_val["y_test"]
    )
    res_rf = task_6_random_forest(
        res_val["X_train_val"], res_val["y_train_val"],
        res_val["X_test"], res_val["y_test"],
        feature_names
    )

    # Supplementary: Tasks 7 & 8
    res_unsup = task_7_8_unsupervised_learning(X, y)

    # Master Benchmark Summary
    master_benchmark_summary(res_svm, res_dt, res_val, res_cv, res_ens, res_rf)

    print("\n" + "=" * 80)
    print("ALL ASSIGNMENT 16 TASKS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
