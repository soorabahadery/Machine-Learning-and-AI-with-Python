import numpy as np                      # For math with arrays (use if you do np.something)
import pandas as pd                     # For working with tables and CSV files
import sklearn as sk                    # For general scikit-learn features (rarely used as 'sk'; most times you import specific functions)
import seaborn as sns                   # For extra-nice statistical charts (use if you do sns.something)
import matplotlib.pyplot as plt         # For making charts and plots
from sklearn import tree                # For drawing and working with decision trees (use tree.plot_tree, etc.)
from sklearn.tree import DecisionTreeClassifier  # For building decision tree models

# Load the toy dataset (with two features and a yes/no label) from a CSV file
tree_df = pd.read_csv('two_classes.csv')

# Show the first 5 rows to see what the data looks like
tree_df.head()

# Plot the data so we can see how the two groups look

# Make a blank chart and set its size
fig, ax = plt.subplots(figsize=(6,6))

# Draw dots for each data point, using color to show which group it belongs to
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Add a legend to say which color means which class
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Name the x and y axes, and give the chart a title
ax.set_xlabel('x1', fontsize='14')
ax.set_ylabel('x2', fontsize='14')
ax.set_title('Synthetic data with two classes', fontsize='15')

# Show the chart on the screen
plt.show()

# This function checks lots of ways to split the data, and gives each split a "Gini score" (lower is better for a decision tree).
def get_total_gini(predictor_values, pred_name, df):
    '''
    Inputs:
        - predictor_values: list of possible numbers to split at
        - pred_name: the column (predictor) to split on
        - df: the table of data
    Output:
        - a list of Gini impurity scores, one for each possible split point
    '''
    total_gini = []
    
    # Try every possible split value
    for val in predictor_values:
    
        # Count class 1 and class 0 in the "left" group (where value is <= val)
        left_1 = np.sum(df.loc[df[pred_name] <= val, 'y'] == 1)
        left_0 = np.sum(df.loc[df[pred_name] <= val, 'y'] == 0)
        
        # Total in left group (protects from dividing by zero)
        N_left = max(1e-5, left_1 + left_0)
        p_left_1 = left_1 / N_left
        p_left_0 = left_0 / N_left

        # Gini score (mixing) for left group
        gini_left = 1 - (p_left_1 ** 2 + p_left_0 ** 2)
    
        # Count class 1 and class 0 in the "right" group (where value is > val)
        right_1 = np.sum(df.loc[df[pred_name] > val, 'y'] == 1)
        right_0 = np.sum(df.loc[df[pred_name] > val, 'y'] == 0)

        # Total in right group (protects from dividing by zero)
        N_right = max(1e-5, right_1 + right_0)
        p_right_1 = right_1 / N_right
        p_right_0 = right_0 / N_right

        # Gini score for right group
        gini_right = 1 - (p_right_1 ** 2 + p_right_0 ** 2)

        # Add up the scores for both groups, adjusted for group size
        N_total = N_left + N_right
        total_gini.append((N_left / N_total) * gini_left + (N_right / N_total) * gini_right)
    
    return total_gini

# Find all the different values in the "x1" column
x1_unique = np.unique(tree_df['x1'].values)

# Find all the different values in the "x2" column
x2_unique = np.unique(tree_df['x2'].values)

# Calculate the Gini impurity score for every possible split using "x1"
x1_total_gini = get_total_gini(x1_unique, 'x1', tree_df)

# Calculate the Gini impurity score for every possible split using "x2"
x2_total_gini = get_total_gini(x2_unique, 'x2', tree_df)

# Draw a chart to show the Gini scores for each way to split using "x1" and "x2"
plt.plot(x1_unique, x1_total_gini, 'r+', label="$x_1$")
plt.plot(x2_unique, x2_total_gini, 'b+', label="$x_2$")
plt.xlabel("Predictor values", fontsize="13")       # Label for the x-axis
plt.ylabel("Total Gini index", fontsize="13")       # Label for the y-axis
plt.title("Total gini indexes for x1 and x2 predictors - first split", fontsize="14")   # Chart title
plt.legend()                                        # Show which color means which predictor
plt.show()                                          # Display the chart

def report_splits(x1_unique, x2_unique, x1_total_gini, x2_total_gini):
    # Figure out which predictor ("x1" or "x2") gives the lowest Gini score for splitting
    best_pred = np.argmin([min(x1_total_gini), min(x2_total_gini)])
    
    # Find where (which value) that lowest Gini score happens
    best_gini_idx = np.argmin([x1_total_gini, x2_total_gini][best_pred])
    
    # The value of "x1" or "x2" where the lowest Gini score is reached
    best_pred_value = [x1_unique, x2_unique][best_pred][best_gini_idx]
    
    # Print out which predictor and value make the best split
    print("The lowest total gini score is achieved when splitting on "+\
          f"{['x1','x2'][best_pred]} at the value {best_pred_value}.")

# Call the function to show the best split choice
report_splits(x1_unique, x2_unique, x1_total_gini, x2_total_gini)

# Find the best place (threshold) to split the data for building the tree
def get_threshold(unique_values, gini_scores):
    # Find where (index) the Gini score is the lowest
    idx = np.argmin(gini_scores)
    
    # If there's a next value after the best one, take the halfway point between them
    if idx + 1 < len(unique_values):
        threshold = (unique_values[idx] + unique_values[idx + 1]) / 2
    else:
        # If it's the last value, just use that one
        threshold = unique_values[idx]
    return threshold

# Find the threshold for splitting "x2" based on its Gini scores
x2_threshold = get_threshold(x2_unique, x2_total_gini)
print(f"Our threshold will be {x2_threshold}")

def get_split_labels(splits):
    '''
    Takes in:
        splits: a list of (predictor name, threshold) pairs
            Example: [('x1', 42), ('x2', 109)]

    Gives back:
        A list of dictionaries, one for each split:
            - Each dictionary says which class is most common on the left and right sides of the split
    '''
    split_labels = []    # Start with an empty list to store results
    region = tree_df     # Work with the whole data table at first
    
    for pred, thresh in splits:
        # For this split, find the most common class (label) on the left and right sides
        region_labels = {
            'left': region.loc[region[pred] <= thresh, 'y'].mode().values[0],
            'right': region.loc[region[pred] > thresh, 'y'].mode().values[0]
        }
        split_labels.append(region_labels)  # Save this result
        
        # For next split, only work with data in the left group
        region = region[region[pred] <= thresh]
    return split_labels

# Example showing how the function works
splits = [('x2', x2_threshold)]
split_labels = get_split_labels(splits)
print('class labels for the children of the root node:', split_labels)

def predict_class(x1, x2, splits):
    # Get the class labels for each side of the splits (from earlier function)
    split_labels = get_split_labels(splits)
    y_hats = []  # This will hold our predicted classes
    
    # Look at every data point, one at a time
    for x1_i, x2_i in zip(x1.ravel(), x2.ravel()):
        obs = {'x1': x1_i, 'x2': x2_i}  # Make a dictionary of values for this point
        
        # Check each split rule for this point
        for n_split, (pred, thresh) in enumerate(splits):
            # If this point goes to the left group
            if obs[pred] <= thresh:
                # If this is the last split, add the left label as the prediction
                if n_split == len(splits)-1:
                    y_hats.append(split_labels[n_split]['left'])
            # If it doesn't go left, it goes right
            else:
                # Add the right label as the prediction
                y_hats.append(split_labels[n_split]['right'])
                break   # Stop checking more splits for this point
    return np.array(y_hats)   # Return all predictions as an array

# Make a chart to show where we split the data

# Set up the blank chart and make it a good size
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the data points; color shows which group they belong to
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Add a legend so people know what the colors mean
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Draw a horizontal line to show the split (our threshold for 'x2')
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()  # Show legend with split line too

# Label the axes and give the chart a title
ax.set_xlabel('x1', fontsize='14')
ax.set_ylabel('x2', fontsize='14')
ax.set_title('The first split of the data', fontsize='16')

# ---- Plot the decision regions ----
# Make a grid of points covering all "x1" and "x2" values
eps = 5  # Small padding around the grid
xx1, xx2 = np.meshgrid(np.arange(tree_df['x1'].min()-eps, tree_df['x1'].max()+eps, 1),
                       np.arange(tree_df['x2'].min()-eps, tree_df['x2'].max()+eps, 1))
# Predict which class each grid point falls into
class_pred = predict_class(xx1, xx2, splits)
# Draw light color shading to show regions where the model predicts each class
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Show the finished chart
plt.show()

# Make a new table with only the rows where "x2" is less than or equal to the split value
first_split_df = tree_df[tree_df['x2'] <= x2_threshold]

# Show the first few rows to see what’s inside
first_split_df.head()

# Find all different values of "x1" and "x2" in our new smaller table
x1_unique_2split = np.unique(first_split_df['x1'].values)
x2_unique_2split = np.unique(first_split_df['x2'].values)

# Calculate the Gini scores for every possible way to split "x1" and "x2" in the new table
tot_gini_x1_2split = get_total_gini(x1_unique_2split, 'x1', first_split_df)
tot_gini_x2_2split = get_total_gini(x2_unique_2split, 'x2', first_split_df)

# Draw a chart showing the Gini scores for each split option
plt.plot(x1_unique_2split, tot_gini_x1_2split, 'r+', label='$x_1$')
plt.plot(x2_unique_2split, tot_gini_x2_2split, 'b+', label='$x_2$')
plt.xlabel("Predictor values", fontsize="13")          # Label for the x-axis
plt.ylabel("Total Gini index", fontsize="13")          # Label for the y-axis
plt.title("Total Gini indexes for x1 and x2 predictors - second split", fontsize="14")  # Chart title
plt.legend()                                           # Show legend for the lines
plt.show()                                             # Display the chart

# Print out which split with the new table and Gini scores is the best
report_splits(x1_unique_2split, x2_unique_2split,
              tot_gini_x1_2split, tot_gini_x2_2split)

# Find the best value to split "x1" in the new table (after first split)
x1_threshold_2split = get_threshold(x1_unique_2split, tot_gini_x1_2split)
x1_threshold_2split

# Make a chart to show both splits on the data

# Set up the chart with the right size
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the data points, colored by which class they are in
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Add a legend to explain class colors
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Draw the first split line (horizontal for x2)
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 0.5, ls='--')

# Draw the second split line (vertical for x1, only goes up to x2 split line)
ax.vlines(x1_threshold_2split, ymin=0, ymax=x2_threshold,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()

# ---- Show color regions where decision tree predicts each class ----
# Predict the class for every grid point using both splits
class_pred = predict_class(xx1, xx2, [('x2', x2_threshold), ('x1', x1_threshold_2split)])
# Shade the regions according to predicted class
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Label the axes and give the chart a title
ax.set_xlabel('x1', fontsize='13')
ax.set_ylabel('x2', fontsize='13')
ax.set_title('First and second splits on the data', fontsize='14')

# Show the final chart
plt.show()

# Make a new table with the rows where "x1" is greater than the second split value (right of the second split)
first_right_intern_node_df  = first_split_df[first_split_df['x1'] > x1_threshold_2split]

# Show the first few rows to see what’s inside
first_right_intern_node_df.head()

# Count how many of each class ("y" value) are in the new table (right of the second split)
first_right_intern_node_df['y'].value_counts()

# Make a new table with the rows where "x1" is less than or equal to the second split value (left of the second split)
second_split_left_df = first_split_df[first_split_df['x1'] <= x1_threshold_2split]

# Count how many of each class ("y" value) are in the new table (left of the second split)
second_split_left_df['y'].value_counts()

# Find the best split for the data left of the second split

# Find all the unique values of "x1" and "x2" in this new region of data
x1_unique_3split = np.unique(second_split_left_df['x1'].values)
x2_unique_3split = np.unique(second_split_left_df['x2'].values)

# Calculate Gini impurity scores for every possible split using "x1" and "x2" in this region
tot_gini_x1_3split = get_total_gini(x1_unique_3split, 'x1', second_split_left_df)
tot_gini_x2_3split = get_total_gini(x2_unique_3split, 'x2', second_split_left_df)

# Draw a chart showing all Gini scores for both predictors (to help pick the best split)
plt.plot(x1_unique_3split, tot_gini_x1_3split, 'r+', label='$x_1$')
plt.plot(x2_unique_3split, tot_gini_x2_3split, 'b+', label='$x_2$')
plt.xlabel("Predictor values", fontsize="13")          # Label for x-axis
plt.ylabel("Total Gini index", fontsize="13")          # Label for y-axis
plt.title("Total Gini indexes for x1 and x2 predictors - third split", fontsize="14")  # Chart title
plt.legend()                                           # Show which color is which predictor
plt.show()                                             # Display the chart

# Print out which split is best for this new third region, using its unique values and Gini scores
report_splits(x1_unique_3split, x2_unique_3split,
              tot_gini_x1_3split, tot_gini_x2_3split)

# Find the best value to split "x2" in the third region (after last split)
x2_threshold_3split = get_threshold(x2_unique_3split, tot_gini_x2_3split)
x2_threshold_3split

# Make a chart showing all three splits on the data

# Set up the chart with the right size
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the data points, colored by their class
scatter = ax.scatter(tree_df['x1'], tree_df['x2'], c=tree_df['y'], cmap='rainbow')

# Add a legend so we know what the colors mean
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.add_artist(legend)

# Draw the first split line (horizontal for x2)
ax.hlines(x2_threshold, xmin=0, xmax=500,
          color ='black', lw = 0.5, ls='--')
# Draw the second split line (vertical for x1, up to x2 threshold)
ax.vlines(x1_threshold_2split, ymin=0, ymax=x2_threshold,
          color ='black', lw = 0.5, ls='--')
# Draw the third split line (horizontal for x2, left of x1 vertical split)
ax.hlines(x2_threshold_3split, xmin=0, xmax=x1_threshold_2split,
          color ='black', lw = 2, ls=':', label='new split')
ax.legend()

# ---- Show color regions where the model predicts each class ----
# Predict the class for every grid point using the three splits
class_pred = predict_class(xx1, xx2, [('x2', x2_threshold),
                                      ('x1', x1_threshold_2split),
                                      ('x2', x2_threshold_3split)])
# Draw shaded regions showing where each class is predicted
plt.contourf(xx1, xx2, class_pred.reshape(xx1.shape), alpha=0.2, zorder=-1, cmap=plt.cm.coolwarm)

# Label the axes and give the chart a title
ax.set_xlabel('x1', fontsize='13')
ax.set_ylabel('x2', fontsize='13')
ax.set_title('First, second, and third splits on the data', fontsize='14')

# Show the finished chart
plt.show()

# Make a table with rows where "x2" is less than or equal to the third split value (final left leaf)
left_final_leaf_df  = second_split_left_df[second_split_left_df['x2'] <= x2_threshold_3split]

# Check how many of each class are in this final leaf (to see if it’s pure)
left_final_leaf_df['y'].value_counts()

# Make a table with rows where "x2" is greater than the third split value (final right leaf)
right_final_leaf_df = second_split_left_df[second_split_left_df['x2'] > x2_threshold_3split]

# Check how many of each class are in this final leaf (to see if it’s pure)
right_final_leaf_df['y'].value_counts()

# Separate our data into predictors (X) and labels (y)
X = tree_df[['x1','x2']]  # X will be the columns x1 and x2
y = tree_df['y']          # y will be the class labels

# Train (fit) a decision tree classifier to the data, limited to depth 3
sklearn_tree = DecisionTreeClassifier(max_depth=3)
sklearn_tree.fit(X, y)  # Fit the tree with predictor data (X) and labels (y)

# Draw a picture of the trained decision tree
tree.plot_tree(sklearn_tree, fontsize=6)
plt.show()  # Show the tree diagram
