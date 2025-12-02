# Import necessary libraries for data analysis and machine learning

import numpy as np                         # For numerical calculations and arrays
import pandas as pd                        # For working with tables/dataframes
import matplotlib.pyplot as plt            # For making charts and plots

from sklearn.metrics import roc_auc_score  # For measuring "AUC" (model quality)
from sklearn.tree import DecisionTreeClassifier       # Decision tree model
from sklearn.ensemble import RandomForestClassifier   # Random Forest model
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.inspection import permutation_importance # Check how important features are

# Command to make plots show up underneath your code (Jupyter notebook magic)
%matplotlib inline

# Read in the "diabetes.csv" dataset as a table (DataFrame)
df = pd.read_csv("diabetes.csv")

# Show the first few rows to see what the data looks like
df.head()

# Set the predictor variables (features) to all columns except "Outcome"
X = df.drop("Outcome", axis=1)

# Set the response variable (target to predict) as "Outcome"
y = df['Outcome']

# Set the seed value to get the same split every time we run the code
seed = 0

# Split the data into training and testing sets (67% train, 33% test)
# Use the seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=seed
)

# Make a default (plain/vanilla) Random Forest classifier
vanilla_rf = RandomForestClassifier(random_state=seed)

# Train (fit) the model using the training data
vanilla_rf.fit(X_train, y_train)

# Get probability predictions for the test set (for ROC AUC calculation)
y_proba = vanilla_rf.predict_proba(X_test)[:, 1]

# Calculate the AUC/ROC score (measures model quality)
auc = np.round(roc_auc_score(y_test, y_proba), 2)

# Print the AUC score for the plain/random forest on the test set
print(f'Plain RF AUC on test set:{auc}')

# Find out the number of features (columns) and samples (rows) in your training data
num_features = X_train.shape[1]  # How many columns (features)
num_samples = X_train.shape[0]   # How many rows (samples)

# Print out the counts to check
num_samples, num_features

%%time
from collections import OrderedDict

# Create a Random Forest classifier:
# - warm_start=True lets you add more trees without restarting
# - oob_score=True gives a quality score using samples not in each tree ("out-of-bag")
# - min_samples_leaf=40 means each leaf must have at least 40 samples
# - max_depth=10 makes trees not too deep
# - random_state=seed ensures reproducibility
clf = RandomForestClassifier(
    warm_start=True,
    oob_score=True,
    min_samples_leaf=40,
    max_depth=10,
    random_state=seed
)

error_rate = {}  # Dictionary to store the OOB error for each tree count

# Choose the range of numbers of trees (n_estimators) to try
min_estimators = 150
max_estimators = 500

# Test different numbers of trees in the forest
for i in range(min_estimators, max_estimators + 1):
    clf.set_params(n_estimators=i)    # Update how many trees to use
    clf.fit(X_train.values, y_train.values)  # Train RF with this number of trees

    # Calculate the Out-of-Bag (OOB) error: (1 - OOB score)
    oob_error = 1 - clf.oob_score_
    error_rate[i] = oob_error         # Store the error for this n_estimators value

%%time
# Plot "OOB error rate" (accuracy) versus number of trees in the Random Forest

xs = []  # List to store n_estimators values (numbers of trees)
ys = []  # List to store the corresponding OOB error rates

# Fill xs and ys with values from the error_rate dictionary
for label, clf_err in error_rate.items():
    xs.append(label)     # Add the number of trees
    ys.append(clf_err)   # Add the error for that number

# Make the plot
plt.plot(xs, ys)                             # Draw line plot of error rate vs number of trees
plt.xlim(min_estimators, max_estimators)     # Limit x axis to the range we tried
plt.xlabel("n_estimators")                   # Label for x axis
plt.ylabel("OOB error rate")                 # Label for y axis
plt.show();                                  # Show the plot!

%%time
from collections import OrderedDict

# Create two Random Forest models with different minimum leaf sizes:
# - First model needs at least 1 sample per leaf
# - Second model needs at least 5 samples per leaf
ensemble_clfs = [
    (1,
        RandomForestClassifier(
            warm_start=True,
            min_samples_leaf=1,
            oob_score=True,
            max_depth=10,
            random_state=seed)),
    (5,
        RandomForestClassifier(
            warm_start=True,
            min_samples_leaf=5,
            oob_score=True,
            max_depth=10,
            random_state=seed))
]

# Prepare a way to store the errors for each min_samples_leaf value
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 80    # Minimum number of trees to test
max_estimators = 500   # Maximum number of trees to test

# For each model ...
for label, clf in ensemble_clfs:
    # Try different numbers of trees from min_estimators to max_estimators
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)               # Set number of trees
        clf.fit(X_train.values, y_train.values)      # Train with current n_estimators

        # Compute OOB error (1 - oob_score)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))     # Save the results

# Now, plot the OOB error for both models
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)                           # Get number of trees and error lists
    plt.plot(xs, ys, label=f'min_samples_leaf={label}') # Plot for this model

plt.xlim(min_estimators, max_estimators)             # Set limits for x axis
plt.xlabel("n_estimators")                           # Label x axis
plt.ylabel("OOB error rate")                         # Label y axis
plt.legend(loc="upper right")                        # Add legend for models
plt.show();                                          # Show the plot!

# Initialize variables to keep track of the lowest error and best settings
err = 100
best_num_estimators = 0

# Loop through the error results for each min_samples_leaf value
for label, clf_err in error_rate.items():
    # For this setting, find the (n_estimators, error) pair with the lowest error
    # If there is a tie, pick the setting with the largest n_estimators
    num_estimators, error = min(clf_err, key=lambda n: (n[1], -n[0]))
    
    # If this error is lower than the best seen so far, update the best tracking variables
    if error < err:
        err = error
        best_num_estimators = num_estimators
        best_leaf = label

# Print out the best settings found
print(f'Optimum num of estimators: {best_num_estimators} \nmin_samples_leaf: {best_leaf}')

### edTest(test_estimators) ### 

# Make a Random Forest model using the best number of trees and min_samples_leaf found above
estimators_rf = RandomForestClassifier(
    n_estimators=best_num_estimators,      # Best number of trees
    random_state=seed,                     # For reproducibility
    oob_score=True,                        # Use OOB score
    min_samples_leaf=best_leaf,            # Best leaf size
    max_features='sqrt'                    # Use sqrt(num_features) for randomness
)

# Train (fit) the optimized Random Forest model on your training data
estimators_rf.fit(X_train, y_train);

# Predict probabilities for the test set (needed for AUC/ROC metric)
y_proba = estimators_rf.predict_proba(X_test)[:, 1]

# Calculate the ROC AUC score, rounded to 2 decimals
estimators_auc = np.round(roc_auc_score(y_test, y_proba), 2)

# Print the AUC score for your tuned ("educated") random forest
print(f'Educated RF AUC on test set:{estimators_auc}')

# Show all the settings (hyperparameters) used by your optimized Random Forest model
estimators_rf.get_params()

%%time
from sklearn.model_selection import GridSearchCV

do_grid_search = True  # Set this to True to run hyperparameter tuning

if do_grid_search:
    # Create a Random Forest model with best values found so far
    rf = RandomForestClassifier(
        n_jobs=-1,                          # Use all CPU cores for speed
        n_estimators=best_num_estimators,    # Number of trees from earlier tuning
        oob_score=True,                      # Use OOB score for validation
        max_features='sqrt',                 # Features per tree
        min_samples_leaf=best_leaf,          # Minimum samples per leaf from earlier
        random_state=seed                    # For reproducibility
    )

    # Dictionary of hyperparameters to try for tuning
    param_grid = {
        'min_samples_split': [2, 5]          # Try min_samples_split = 2 and 5
    }
    
    scoring = {'AUC': 'roc_auc'}             # Use AUC/ROC score for selecting best model
    
    # Set up grid search to try each combination of parameters
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        scoring=scoring, 
        refit='AUC',                # Pick best model using AUC score
        return_train_score=True,    # Store training scores
        n_jobs=-1                   # Use all processors
    )
    
    # Fit GridSearchCV to training data to find the best settings
    results = grid_search.fit(X_train, y_train)
    
    # Print out all the hyperparameters of the best model found
    print(results.best_estimator_.get_params())
    
    # Save the best model
    best_rf = results.best_estimator_

    # Calculate ROC AUC score for this best model on the test set
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    auc = np.round(roc_auc_score(y_test, y_proba), 2)
    print(f'GridSearchCV RF AUC on test set:{auc}')
