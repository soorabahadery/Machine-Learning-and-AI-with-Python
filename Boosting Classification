# Import necessary libraries for boosting classification and visualizations

import numpy as np                       # For arrays and calculations
import pandas as pd                      # For data tables
import seaborn as sns                    # For pretty statistical plots
import matplotlib.pyplot as plt           # For charts and graphs

from helper import plot_decision_boundary # For plotting model decision boundaries (custom function in your helper.py)
from matplotlib.colors import ListedColormap # For coloring plots by class

from sklearn.tree import DecisionTreeClassifier  # Basic decision tree for classification
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost: classic boosting classifier

%matplotlib inline    # Show plots inside Jupyter notebook
sns.set_style('white') # Make Seaborn plots use a clean white background

# Read the dataset "boostingclassifier.csv" into a pandas DataFrame
df = pd.read_csv("boostingclassifier.csv")

# Set 'latitude' and 'longitude' as the predictor variables (features for classification)
X = df[['latitude', 'longitude']].values

# Set 'landtype' as the response variable (target/class to predict)
y = df['landtype'].values

### edTest(test_response) ###
# Update the class labels for AdaBoost
# Replace all 0s in y with -1, while 1s stay as 1 (so labels are -1 and 1)
y = np.where(y == 0, -1, 1)

# AdaBoost algorithm implementation from scratch

def AdaBoost_scratch(X, y, M=10):
    '''
    X: data matrix of predictors (features)
    y: response variable (target labels: -1, 1)
    M: number of estimators (how many stumps/trees to boost)
    '''

    # Initialization of helper lists for storing model and data for each boosting round
    N = len(y)                              # Number of data points
    estimator_list = []                     # Where we'll store each decision tree
    y_predict_list = []                     # Predictions from each round
    estimator_error_list = []               # Error for each round
    estimator_weight_list = []              # Gives importance to each tree
    sample_weight_list = []                 # Stores sample weights for each round

    # Start with equal weights for all data points
    sample_weight = np.ones(N) / N
    
    # Store the initial weights
    # We use .copy() so each round's weights won't get overwritten later!
    sample_weight_list.append(sample_weight.copy())

    # Repeat for the number of boosting rounds
    # We can use '_' as a throwaway variable since we don't need the round number
    for _ in range(M):
        # Create a basic decision stump (depth = 1)
        estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
        
        # Fit the stump using current sample weights
        estimator.fit(X, y, sample_weight=sample_weight)
        
        # Predict all labels with this stump
        y_predict = estimator.predict(X)

        # Find which predictions were wrong (makes a 0/1 array)
        incorrect = (y_predict != y).astype(int)

        # Figure out the weighted error (more heavily weighted points matter more)
        estimator_error = np.average(incorrect, weights=sample_weight)
        
        # Calculate the weight for this stump (higher if it was good, lower if bad)
        estimator_weight = 0.5 * np.log((1 - estimator_error) / estimator_error)

        # Update sample weights (make hard-to-classify points weightier for next tree)
        sample_weight *= np.exp(-estimator_weight * y * y_predict)

        # Normalize them so they add up to 1
        sample_weight /= np.sum(sample_weight)

        # Save all values from this round
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert lists to numpy arrays
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # For the final prediction: add up weighted votes of all trees, then use np.sign to pick class
    # np.sign will turn positive sums into +1 and negative sums into -1, for the final class
    preds = (np.array([np.sign((y_predict_list[:,point] * estimator_weight_list).sum()) for point in range(N)]))
    
    # Return everything so you can inspect the boosting process, weights, and predictions
    return estimator_list, estimator_weight_list, sample_weight_list, preds

### edTest(test_adaboost) ###
# Run AdaBoost from scratch on your data, using M=9 stumps
estimator_list, estimator_weight_list, sample_weight_list, preds = AdaBoost_scratch(X, y, M=9)

# Calculate accuracy: How many predictions match the true labels?
accuracy = np.mean(preds == y)

# Print the accuracy as a decimal with 3 digits
print(f'accuracy: {accuracy:.3f}')

# Plot the decision boundaries for each AdaBoost stump (iteration)
fig = plt.figure(figsize=(16, 16))  # Make a big grid figure

for m in range(0, 9):
    # Add a subplot for each of the 9 stumps
    fig.add_subplot(3, 3, m + 1)
    
    # Calculate point sizes based on sample weights for this round
    s_weights = (sample_weight_list[m, :] / sample_weight_list[m, :].sum()) * 300
    
    # Use your helper function to plot the decision boundary for this estimator
    plot_decision_boundary(estimator_list[m], X, y, N=50, scatter_weights=s_weights, counter=m)
    
    plt.tight_layout()  # Avoid overlapping subplots

# Use sklearn's AdaBoost to plot the combined/classifier's decision boundary

# Make an AdaBoost model using stumps (max_depth=1), with 9 estimators
boost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Base estimator: tree stump
    n_estimators=9                                  # Number of boosting rounds
)

# Fit AdaBoost on ALL the data
boost.fit(X, y)

# Plot the overall decision boundary using your helper function
plot_decision_boundary(boost, X, y, N=50)

plt.title('AdaBoost Decision Boundary', fontsize=16)
plt.show()
