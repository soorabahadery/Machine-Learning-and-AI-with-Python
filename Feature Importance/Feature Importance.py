# Import key libraries for data and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")  # Hide warnings for clean output

# Helper function: Plot impurity-based (default) feature importances of two models
def plot_feature_importance(model1, model2, X, y):
    # model1: usually a single tree (e.g., DecisionTreeClassifier)
    # model2: usually a random forest (ensemble of trees)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ------ Single Tree Plot ------
    tree_importance_sorted_idx = np.argsort(model1.feature_importances_)
    tree_indices = np.arange(0, len(model1.feature_importances_)) + 0.5

    ax1.barh(tree_indices, 
            model1.feature_importances_[tree_importance_sorted_idx], 
            height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X.columns[tree_importance_sorted_idx], fontsize=12)
    ax1.set_ylim((0, len(model1.feature_importances_)))
    ax1.set_xlabel("Impurity Based Feature Importance", fontsize=16)
    ax1.set_title("Single Tree", fontsize=18)
    
    # ------ Random Forest Plot ------
    tree_importance_sorted_idx = np.argsort(model2.feature_importances_)
    tree_indices = np.arange(0, len(model2.feature_importances_)) + 0.5
    difference = model2.feature_importances_ - model1.feature_importances_
    difference = difference[tree_importance_sorted_idx]

    ax2.barh(tree_indices, model2.feature_importances_[tree_importance_sorted_idx], 
            height=0.7, color='#EFAEA4')
    # Annotate with difference between RF and single tree
    for index, value in enumerate(model2.feature_importances_[tree_importance_sorted_idx]):
        ax2.text(value, index + 0.3, f" {str(round(difference[index],3))}", fontsize=14)
    ax2.set_yticks(tree_indices)
    ax2.set_yticklabels(X.columns[tree_importance_sorted_idx], fontsize=12)
    ax2.set_ylim((0, len(model2.feature_importances_)))
    maxlim = max(model2.feature_importances_)
    ax2.set_xlim(0, maxlim + 0.02)
    ax2.set_xlabel("Impurity Based Feature Importance", fontsize=16)
    ax2.set_title("Random Forest", fontsize=18)

    fig.tight_layout()
    plt.show()


# Helper function: Plot permutation feature importances for two models
def plot_permute_importance(result1, result2, X, y):
    # result1 and result2 are outputs of permutation_importance(...)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ------ Single Tree Permutation Importance ------
    tree_importance_sorted_idx = np.argsort(result1.importances_mean)
    tree_indices = np.arange(0, len(result1.importances_mean)) + 0.5

    ax1.barh(tree_indices, result1.importances_mean[tree_importance_sorted_idx], 
            height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X.columns[tree_importance_sorted_idx], fontsize=12)
    ax1.set_ylim((0, len(result1.importances_mean)))
    ax1.set_xlabel("Permutation Feature Importance", fontsize=16)
    ax1.set_title("Single Tree", fontsize=18)
    
    # ------ Random Forest Permutation Importance ------
    tree_importance_sorted_idx2 = np.argsort(result2.importances_mean)
    tree_indices2 = np.arange(0, len(result2.importances_mean)) + 0.5
    difference = result2['importances_mean'] - result1['importances_mean']
    difference = difference[tree_importance_sorted_idx]

    ax2.barh(tree_indices2, result2.importances_mean[tree_importance_sorted_idx2], 
            height=0.7, color='#EFAEA4')
    # Annotate with difference between RF and single tree
    for index, value in enumerate(result2.importances_mean[tree_importance_sorted_idx2]):
        ax2.text(value, index + 0.3, f" {str(round(difference[index],3))}", fontsize=14)
    ax2.set_yticks(tree_indices2)
    ax2.set_yticklabels(X.columns[tree_importance_sorted_idx2], fontsize=12)
    ax2.set_ylim((0, len(result2.importances_mean)))
    ax2.set_xlabel("Permutation Feature Importance", fontsize=16)
    maxlim = max(result2.importances_mean)
    ax2.set_xlim(0, maxlim + 0.015)
    ax2.set_title("Random Forest", fontsize=18)

    fig.tight_layout()
    plt.show()
