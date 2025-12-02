# Import main Python packages needed for data and machine learning

import numpy as np                      # For math and arrays
import pandas as pd                     # For tables and dataframes
import matplotlib.pyplot as plt         # For drawing charts and plots
import dtreeviz                         # For fancy decision tree visualizations
from sklearn.metrics import accuracy_score               # For checking model accuracy
from sklearn.tree import DecisionTreeClassifier          # For decision tree classification models
from sklearn.model_selection import train_test_split     # To divide data into train/test sets
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier  # For Random Forest and Bagging models

# Plotting command to show charts right under code in Jupyter
%matplotlib inline

# These lines make it easy to print markdown-formatted text in Jupyter
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# Read in the "diabetes.csv" dataset as a table (DataFrame)
df = pd.read_csv("diabetes.csv")

# Show the first few rows to see what the data looks like
df.head()

# Set the predictor variables (features) as all columns except "Outcome"
X = df.drop("Outcome", axis=1)

# Set the response variable (target/class) as the "Outcome" column
y = df["Outcome"]

# Fix a random seed for reproducibility
random_state = 144

# Split the data into training (80%) and validation (20%) sets,
# using the random seed above, and keep balanced class proportions (stratify)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=random_state, stratify=y
)

# Set how deep each decision tree can be (limit = 20)
max_depth = 20

# Choose how many decision trees will be used in Bagging (1000)
n_estimators = 1000

# Make a base decision tree model with the chosen depth and random seed
basemodel = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

# Make the Bagging classifier:
# - Use the decision tree as the "base model"
# - Use 1000 trees
# - Use the same random seed for reproducibility
bagging = BaggingClassifier(
    estimator=basemodel,
    n_estimators=n_estimators,
    random_state=random_state
)

# Train (fit) the Bagging model using the training data
bagging.fit(X_train, y_train)

# Use the trained Bagging model to predict outcomes for the validation set
predictions = bagging.predict(X_val)

# Calculate how accurate the predictions are (compare to actual values)
acc_bag = round(accuracy_score(predictions, y_val), 2)

# Print the validation accuracy
print(f'For Bagging, the accuracy on the validation set is {acc_bag}')

# Make a Random Forest classifier:
# - Use 1000 trees
# - Each tree has a maximum depth of 20
# - Use the same random seed for reproducibility
random_forest = RandomForestClassifier(
    n_estimators=1000,
    max_depth=max_depth,
    random_state=random_state
)

# Train (fit) the Random Forest model using the training data
random_forest.fit(X_train, y_train)

# Use the trained Random Forest model to predict outcomes for the validation set
predictions = random_forest.predict(X_val)

# Calculate how accurate the predictions are (compare to actual values)
acc_rf = round(accuracy_score(predictions, y_val), 2)

# Print the validation accuracy
print(f'For Random Forest, the accuracy on the validation set is {acc_rf}')

# Helper code to visualize decision trees from the Bagging ensemble

# Set each tree to depth 3 so they're easier to visualize
max_depth = 3

# Make a base decision tree model (low depth and random seed)
basemodel = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

# Make a Bagging model using the shallow decision tree and 1000 trees
bagging = BaggingClassifier(estimator=basemodel, n_estimators=1000)

# Train (fit) the Bagging model using the training data
bagging.fit(X_train, y_train)

# Pick two random trees from the Bagging ensemble
bagvati1 = bagging.estimators_[0]     # The first tree
bagvati2 = bagging.estimators_[100]   # The 101st tree

# Visualize the first tree from the Bagging ensemble using dtreeviz

vizA = dtreeviz.model(
    bagvati1,                               # The trained decision tree to visualize
    df.iloc[:, :8],                        # Feature data (first 8 columns)
    df.Outcome,                            # Target variable (Outcome column)
    feature_names = df.columns[:8],        # Feature names
    target_name = 'Diabetes',              # Label for the target variable
    class_names = ['No', 'Yes'],           # Names for the classes
)

# Set parameters for the visualization look (font, size, scale)
viz_params = {
    'fontname': 'monospace',
    'label_fontsize': 18,
    'ticks_fontsize': 16,
    'scale': 1.4
}

# Show the visualization with the chosen settings
vizA.view(**viz_params);

# Visualize the second (101st) tree from the Bagging ensemble using dtreeviz

vizB = dtreeviz.model(
    bagvati2,                              # The selected decision tree to visualize
    df.iloc[:, :8],                        # Feature data (first 8 columns)
    df.Outcome,                            # Target variable (Outcome column)
    feature_names = df.columns[:8],        # Feature names
    target_name = 'Diabetes',              # Label for the target variable
    class_names = ['No', 'Yes']            # Names for the classes
)

# Show the visualization using the same visual style as before
vizB.view(**viz_params);

# Helper code to visualize trees from a Random Forest ensemble

# Reduce each tree’s depth to 3 for easier viewing
max_depth = 3

# Create the Random Forest model:
# - Each tree has depth limited to 3
# - Use 1000 trees in total
# - Random seed is set for reproducibility
# - max_features="sqrt" so each tree sees only a random subset of features (the "random" part!)
random_forest = RandomForestClassifier(
    max_depth=max_depth,
    random_state=random_state,
    n_estimators=1000,
    max_features="sqrt"
)

# Train (fit) the Random Forest model on the training data
random_forest.fit(X_train, y_train)

# Choose two trees from the forest to visualize
forestvati1 = random_forest.estimators_[0]    # The first tree
forestvati2 = random_forest.estimators_[100]  # The 101st tree

# Visualize the first tree from the Random Forest ensemble using dtreeviz

vizC = dtreeviz.model(
    forestvati1,                         # The selected decision tree to visualize
    df.iloc[:, :8],                      # Feature data (first 8 columns)
    df.Outcome,                          # Target variable (Outcome column)
    feature_names = df.columns[:8],      # Feature names
    target_name = 'Diabetes',            # Label for the target variable
    class_names = ['No', 'Yes']          # Names for the classes
)

# Show the visualization using the style settings from earlier
vizC.view(**viz_params)

# Visualize the second (101st) tree from the Random Forest ensemble using dtreeviz

vizD = dtreeviz.model(
    forestvati2,                        # The selected decision tree to visualize
    df.iloc[:, :8],                     # Feature data (first 8 columns)
    df.Outcome,                         # Target variable (Outcome column)
    feature_names = df.columns[:8],     # Feature names
    target_name = 'Diabetes',           # Label for the target variable
    class_names = ['No', 'Yes']         # Names for the classes
)

# Show the visualization using the given style settings
vizD.view(**viz_params)
