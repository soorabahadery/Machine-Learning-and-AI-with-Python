# Make charts show up in our notebook, right below the code
%matplotlib inline

# Import libraries we need for our project
import numpy as np                   # For math and arrays
import pandas as pd                  # For tables and dataframes
from sklearn import metrics          # For machine learning metrics
import scipy.optimize as opt         # For optimization (math, fitting, etc.)
import matplotlib.pyplot as plt      # For drawing charts and plots
from sklearn.metrics import accuracy_score        # For checking how accurate our model is
from sklearn.tree import DecisionTreeClassifier   # For decision tree classification
from sklearn.model_selection import train_test_split  # To split our data into training and test sets

# Set up colors for plotting decision regions later
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#F7345E','#80C3BD'])    # Bold colors for plotting classes
cmap_light = ListedColormap(['#FFF4E5','#D2E3EF'])   # Light colors for background regions

# Read in the "agriland.csv" file and store it as a table called df
df = pd.read_csv('agriland.csv')

# Show the first few rows to take a quick look at the data
# (The latitude & longitude values are already normalized)
df.head()

# Use the "latitude" and "longitude" columns as the predictor variables (features)
X = df[['latitude', 'longitude']].values

# Use the "land_type" column as the response variable (target/class)
y = df['land_type'].values

# Split the data into training (80%) and testing (20%) sets, using random state 44 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Choose how deep the decision tree can grow (here, max depth is 3)
max_depth = 3

# Make a decision tree classifier:
# - Limit depth to the value above
# - Use random_state 44 for reproducibility
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=44)

# Train (fit) the decision tree classifier model on our training data
clf.fit(X_train, y_train)

# Use the trained decision tree to predict land types for the test data
prediction = clf.predict(X_test)

# Calculate how many predictions were correct (accuracy) for the test data
single_acc = accuracy_score(y_test, prediction)

# Print out the accuracy as a percentage
print(f'Single tree Accuracy is {single_acc*100}%')

# Make predictions on data by combining several decision trees (bagging)
# Inputs:
#   - X_train, y_train: Training data and labels
#   - X_to_evaluate: Data to predict on
#   - num_bootstraps: How many decision trees (bagging rounds) to use
# Output:
#   - An array of predicted classes ("majority vote") for X_to_evaluate

def prediction_by_bagging(X_train, y_train, X_to_evaluate, num_bootstraps):
    
    # List to store each tree's predictions
    predictions = []
    
    # Train num_bootstraps number of trees
    for i in range(num_bootstraps):
        
        # Pick random samples (with replacement) from train data
        resample_indexes = np.random.choice(np.arange(y_train.shape[0]), size=y_train.shape[0])
        
        # Make the bootstrapped training set
        X_boot = X_train[resample_indexes]
        y_boot = y_train[resample_indexes]
        
        # Make and train a decision tree (same depth and random_state)
        clf = DecisionTreeClassifier(max_depth=3, random_state=44)
        clf.fit(X_boot, y_boot)
        
        # Predict class for the evaluation data
        pred = clf.predict(X_to_evaluate)
        
        # Save this tree's predictions
        predictions.append(pred)
    
    # Turn list of predictions into a 2D array (each row: one tree's predictions)
    predictions = np.array(predictions)
    
    # Do a majority vote for each sample: choose the class that appears most
    # For each column (each sample), count the most frequent value
    from scipy.stats import mode
    majority_vote = mode(predictions, axis=0).mode.flatten()
    
    # Return the final prediction for each sample (majority vote)
    return majority_vote

# Set how many decision trees (bootstraps) we want to use
num_bootstraps = 200

# Get predictions from our bagging function using the training data and test data
y_pred = prediction_by_bagging(X_train, y_train, X_test, num_bootstraps=num_bootstraps)

# Check how many predictions agree with the actual test labels (accuracy)
bagging_accuracy = accuracy_score(y_test, y_pred)

# Print out the bagging accuracy as a percentage
print(f'Accuracy with Bootstrapped Aggregation is  {bagging_accuracy*100}%')

# See how accuracy changes as we use more decision trees in bagging

n = np.linspace(1, 250, 250).astype(int)   # List of tree counts from 1 to 250
acc = []    # Will store the accuracy for each number of trees

# For each number of trees, get bagged predictions and compute accuracy
for n_i in n:
    preds = prediction_by_bagging(X_train, y_train, X_test, n_i)
    # Check how many predictions are correct by comparing with test labels
    accuracy = np.mean(preds == y_test)
    acc.append(accuracy)

# Draw the plot
plt.figure(figsize=(10,8))
plt.plot(n, acc, alpha=0.7, linewidth=3, color='#50AEA4', label='Model Prediction')

plt.title('Accuracy vs. Number of trees in Bagging', fontsize=24)
plt.xlabel('Number of trees', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.show()

# Plot decision boundaries and data for three different decision tree depths

fig, axes = plt.subplots(1, 3, figsize=(16, 4))  # Make 3 plots side by side

# List of three different tree depths to compare
max_depth = [2, 5, 100]

# Number of bootstrapped samples to plot for each depth
numboot = 100

# Go through each depth/axis
for index, ax in enumerate(axes):

    for i in range(numboot):
        # Make a new version of the data by sampling with replacement (bootstrapping)
        df_new = df.sample(frac=1, replace=True)
        y = df_new.land_type.values         # Get the class labels
        X = df_new[['latitude', 'longitude']].values   # Get the features

        # Make and train a decision tree for this depth
        dtree = DecisionTreeClassifier(max_depth=max_depth[index])
        dtree.fit(X, y)

        # Scatter plot the sampled data points, color by class
        ax.scatter(X[:, 0], X[:, 1], c=y-1, s=50, alpha=0.5, edgecolor="k", cmap=cmap_bold) 
        
        # Setup a grid for plotting the decision boundaries
        plot_step_x1 = 0.1
        plot_step_x2 = 0.1
        x1min, x1max = X[:,0].min(), X[:,0].max()
        x2min, x2max = X[:,1].min(), X[:,1].max()
        x1, x2 = np.meshgrid(np.arange(x1min, x1max, plot_step_x1), np.arange(x2min, x2max, plot_step_x2))
        Xplot = np.c_[x1.ravel(), x2.ravel()]  # Flatten grid for prediction

        # Predict the class at each grid point
        y_grid = dtree.predict(Xplot)
        y_grid = y_grid.reshape(x1.shape)
        cs = ax.contourf(x1, x2, y_grid, alpha=0.02)   # Lightly plot decision boundaries
        
    # Label the axes and add a title showing tree depth
    ax.set_xlabel('Latitude', fontsize=14)
    ax.set_ylabel('Longitude', fontsize=14)
    ax.set_title(f'Max depth = {max_depth[index]}', fontsize=20)
