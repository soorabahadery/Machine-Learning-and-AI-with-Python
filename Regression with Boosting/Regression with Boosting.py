# Import necessary libraries for regression and data analysis

import itertools                       # For advanced looping, if needed
import numpy as np                     # For numerical calculations and arrays
import pandas as pd                    # For data tables (DataFrame)
import matplotlib.pyplot as plt         # For making charts and plots

from sklearn.ensemble import BaggingRegressor           # Bagging (ensemble) regression
from sklearn.tree import DecisionTreeRegressor          # Single decision tree regressor
from sklearn.metrics import mean_squared_error          # For checking how good predictions are
from sklearn.model_selection import train_test_split    # To split data into train/test sets
from sklearn.ensemble import GradientBoostingRegressor  # Boosting (ensemble) regression

%matplotlib inline                       # Show plots inside Jupyter notebook cells

# Read the airquality.csv file into a table (DataFrame)
df = pd.read_csv("airquality.csv")

# Remove rows where the 'Ozone' value is missing (to clean the data)
df = df[df.Ozone.notna()]

# Show the first few rows to quickly look at what the data looks like
df.head()

# Set 'Ozone' as the predictor (input feature) and 'Temp' as the response (target to predict)
x, y = df['Ozone'].values, df['Temp'].values

# Sort the data by increasing Ozone value, so plots look nice and smooth
x, y = list(zip(*sorted(zip(x, y))))
x, y = np.array(x).reshape(-1, 1), np.array(y)

# Create a decision tree with max_depth=1 ("stump": only splits once, very simple)
basemodel = DecisionTreeRegressor(max_depth=1)

# Train (fit) the stump on all the ozone/temp data
basemodel.fit(x, y)

# Use the stump to predict temperatures for every ozone value in our data
y_pred = basemodel.predict(x)

# Plot the data and predictions of your first decision tree stump
plt.figure(figsize=(10,6))  # Make a big, clear plot

xrange = np.linspace(x.min(), x.max(), 100)  # (Not strictly needed here unless for smooth curves)

# Plot the actual data points (Ozone vs Temperature)
plt.plot(x, y, 'o', color='#EFAEA4', markersize=6, label="True Data")

# Plot the predictions from your tree stump
plt.plot(x, y_pred, alpha=0.7, linewidth=3, color='#50AEA4', label='First Tree')

plt.xlabel("Ozone", fontsize=16)         # Label for x axis
plt.ylabel("Temperature", fontsize=16)   # Label for y axis
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=12)      # Show labels for each curve

plt.show()   # Display the plot!

# Make a plot to show the data, model predictions, and the residuals (errors)

plt.figure(figsize=(10,6))  # Make a big, clear plot

# Plot the true data: Ozone vs Temperature
plt.plot(x, y, 'o', color='#EFAEA4', markersize=6, label="True Data")

# Plot the residuals: difference between actual and predicted temperature (errors)
plt.plot(x, residuals, '.-', color='#faa0a6', markersize=6, label="Residuals")

# Draw a dashed line at zero to show where errors would be zero
plt.plot([x.min(), x.max()], [0, 0], '--')

plt.xlim()  # Use the default limits

# Plot the predicted values from your stump
plt.plot(x, y_pred, alpha=0.7, linewidth=3, color='#50AEA4', label='First Tree')

plt.xlabel("Ozone", fontsize=16)        # Label for x axis
plt.ylabel("Temperature", fontsize=16)  # Label for y axis
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='center right', fontsize=12)  # Place the legend nicely

plt.show()  # Display the plot!

### edTest(test_fitted_residuals) ###

# Create a new decision tree with depth=1 (stump)
dtr = DecisionTreeRegressor(max_depth=1)

# Train this stump to fit the residuals (what the first tree got wrong)
dtr.fit(x, residuals)

# Predict the residuals for all data points using this new stump
y_pred_residuals = dtr.predict(x)

# Plot the original data, residuals, first tree predictions, and the tree fit to residuals

plt.figure(figsize=(10,6))  # Large, clear plot

# Show the actual data points (Ozone vs Temperature)
plt.plot(x, y, 'o', color='#EFAEA4', markersize=6, label="True Data")

# Show the residuals (difference between actual and predicted)
plt.plot(x, residuals, '.-', color='#faa0a6', markersize=6, label="Residuals")

# Dashed line at zero to highlight the baseline for residuals
plt.plot([x.min(), x.max()], [0, 0], '--')

plt.xlim()  # Use default axis limits

# The prediction of the first tree (stump)
plt.plot(x, y_pred, alpha=0.7, linewidth=3, color='#50AEA4', label='First Tree')

# The prediction of the second tree (fitted just on the residuals)
plt.plot(x, y_pred_residuals, alpha=0.7, linewidth=3, color='red', label='Residual Tree')

plt.xlabel("Ozone", fontsize=16)         # x axis label
plt.ylabel("Temperature", fontsize=16)   # y axis label
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='center right', fontsize=12)  # Legend for lines

plt.show()  # Display the finished plot!

### edTest(test_new_pred) ###

# Set a learning rate for the second tree (how much of the residual tree to add)
lambda_ = 0.8

# Combine predictions: first tree + 0.8 * second tree (which fits the residuals)
y_pred_new = y_pred + lambda_ * y_pred_residuals

# Plot everything: true data, residuals, first tree, residual tree, and boosted prediction

plt.figure(figsize=(10, 8))  # Make a big plot

# Plot true data points (Ozone vs Temperature)
plt.plot(x, y, 'o', color='#EFAEA4', markersize=6, label="True Data")

# Plot residuals (errors)
plt.plot(x, residuals, '.-', color='#faa0a6', markersize=6, label="Residuals")

# Dashed horizontal zero-line as reference for residuals
plt.plot([x.min(), x.max()], [0, 0], '--')

plt.xlim()  # Use default axis limits

# Plot the prediction from the first tree
plt.plot(x, y_pred, alpha=0.7, linewidth=3, color='#50AEA4', label='First Tree')

# Plot the prediction from the tree fitted to residuals
plt.plot(x, y_pred_residuals, alpha=0.7, linewidth=3, color='red', label='Residual Tree')

# Plot the boosted prediction (first tree + lambda * residual tree)
plt.plot(x, y_pred_new, alpha=0.7, linewidth=3, color='k', label='Boosted Tree')

plt.xlabel("Ozone", fontsize=16)        # x axis label
plt.ylabel("Temperature", fontsize=16)  # y axis label
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='center right', fontsize=12)

plt.show()  # Show the plot!

# Split the data into training and test sets
# 80% training, 20% test, using random_state for reproducibility

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=102
)

### edTest(test_boosting) ###

# Set the "learning rate" for boosting (how strongly each tree corrects errors)
l_rate = 0.8

# Create a boosting regression model:
# - 1000 trees (each tries to improve on previous ones)
# - Each tree is very simple (max_depth=1)
# - Uses the learning rate you chose
boosted_model = GradientBoostingRegressor(
    n_estimators=1000,         # Use 1000 trees in this ensemble
    max_depth=1,               # Each tree is just a "stump"
    learning_rate=l_rate,      # Learning rate for boosting
    random_state=42            # For reproducible results
)

# Train the boosting model on your training data
boosted_model.fit(x_train, y_train)

# Use the trained boosting model to predict temperature on your test data
y_pred = boosted_model.predict(x_test)

# Specify how many trees ("bootstraps") you want in your ensemble
num_bootstraps = 30

# Set the maximum depth for each decision tree in your bagging model
max_depth = 100

# Define the Bagging Regressor Model:
# - Uses Decision Tree as the base model, with max depth 100
# - Builds 30 trees (num_bootstraps)
# - Each tree trains on a random 80% of the data (max_samples=.8)
# - random_state keeps results reproducible
model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=max_depth),  # Base tree for bagging
    n_estimators=num_bootstraps,            # Number of bagged trees
    max_samples=0.8,                        # Fraction of data for each tree
    random_state=3                          # Reproducible split
)

# Train (fit) the bagging model on your training data
model.fit(x_train, y_train)

# Plot the predictions from both Bagging and Boosting models alongside the true data

plt.figure(figsize=(10, 8))  # Make a large plot for clear comparison

# Create a smooth spread of Ozone values for making prediction curves
xrange = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

# Predict Temperature for all ozone values in xrange using Boosting and Bagging models
y_pred_boost = boosted_model.predict(xrange)
y_pred_bag = model.predict(xrange)

# Plot the actual data points (Ozone vs Temperature)
plt.plot(x, y, 'o', color='#EFAEA4', markersize=6, label="True Data")

plt.xlim()  # Use default x limits

# Plot the Boosting model's predicted curve
plt.plot(xrange, y_pred_boost, alpha=0.7, linewidth=3, color='#77c2fc', label='Boosting')

# Plot the Bagging model's predicted curve
plt.plot(xrange, y_pred_bag, alpha=0.7, linewidth=3, color='#50AEA4', label='Bagging')

plt.xlabel("Ozone", fontsize=16)        # x axis label
plt.ylabel("Temperature", fontsize=16)  # y axis label
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=12)      # Choose best location for legend

plt.show()  # Show the plot!

### edTest(test_mse) ###

# Calculate the mean squared error (MSE) for Boosting predictions on the test data
boost_mse = mean_squared_error(y_test, y_pred)

# Print out the result so you can see how well your boosting model performed
print("The MSE of the Boosting model is", boost_mse)

# Calculate the mean squared error (MSE) for Bagging predictions on the test data
bag_mse = mean_squared_error(y_test, model.predict(x_test))

# Print the bagging model's MSE so you can compare it to boosting
print("The MSE of the Bagging model is", bag_mse)
