# Import libraries we need for this project

import itertools                # For making combinations or product sets
import numpy as np              # For math and arrays
import pandas as pd             # For tables and dataframes

from numpy import std           # To calculate standard deviation
from numpy import mean          # To calculate average (mean)

import matplotlib.pyplot as plt # For drawing charts and plots

from sklearn.datasets import make_regression      # To quickly make fake regression data
from sklearn.ensemble import BaggingRegressor     # For bagging regression models (like random forests)
from sklearn.tree import DecisionTreeRegressor    # For decision tree regression
from sklearn.metrics import mean_squared_error    # For checking how good our predictions are (MSE)
from sklearn.model_selection import train_test_split # To split our data into training and test sets

%matplotlib inline      # So plots show up right inside our notebook

# Read in the "airquality.csv" file and store it in a table called df
df = pd.read_csv("airquality.csv", index_col=0)

# Show the first 10 rows to see what the data looks like
df.head(10)

# Remove any rows where the "Ozone" value is missing (NaN)
df = df[df.Ozone.notna()]

# Use "Ozone" values as the predictor variable (x)
x = df[['Ozone']].values

# Use "Temp" (temperature) values as the response variable (y)
y = df['Temp']

# Split the data into training (80%) and testing (20%) sets, using random state 102 for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=102)

# Set how many bootstrapped models to use (here 30)
num_bootstraps = 30

# Set the maximum depth for each decision tree (here 3)
max_depth = 3

# Make a bagging regressor model:
# - It will use decision trees (max depth set above) as the basic model
# - It will use 30 trees (from num_bootstraps)
# - We set random_state so results are always the same
model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=max_depth),
    n_estimators=num_bootstraps,
    random_state=102
)

# Train (fit) the bagging regressor model on our training data
model.fit(x_train, y_train)

# Draw a chart to see how each of the individual decision trees (estimators) predicts, and how the whole bagging model does

plt.figure(figsize=(10,8))   # Set the size of the plot

# Create evenly spaced Ozone values for prediction (for smooth lines)
xrange = np.linspace(x.min(), x.max(), 80).reshape(-1,1)

# Plot the training data points (Ozone vs. Temp)
plt.plot(x_train, y_train, 'o', color='#CC1100', markersize=6, label="Train Data")

# Plot the test data points (Ozone vs. Temp)
plt.plot(x_test, y_test, 'o', color='#241571', markersize=6, label="Test Data")

plt.xlim()  # Keep default x-axis limits

# Plot the predictions from each individual estimator (light lines)
for i in model.estimators_:
    y_pred1 = i.predict(xrange)
    plt.plot(xrange, y_pred1, alpha=0.5, linewidth=0.5, color='#ABCCE3')

# Plot the prediction from the last estimator again, but bolder with label
plt.plot(xrange, y_pred1, alpha=0.6, linewidth=1, color='#ABCCE3', label="Prediction of Individual Estimators")

# Plot the combined prediction from the whole bagging model (thick bright line)
y_pred = model.predict(xrange)
plt.plot(xrange, y_pred, alpha=0.7, linewidth=3, color='#50AEA4', label='Model Prediction')

# Label axes and style the chart
plt.xlabel("Ozone", fontsize=16)
plt.ylabel("Temperature", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best', fontsize=12)

# Show the complete plot
plt.show()

# Use the first decision tree in our bagging model to predict temp values on the test data
y_pred1 = model.estimators_[0].predict(x_test)

# Calculate and print the test mean squared error (MSE) for this estimator
print("The test MSE of one estimator in the model is", round(mean_squared_error(y_test, y_pred1), 2))

# Predict temperature values on the test data using the whole bagging model
y_pred = model.predict(x_test)

# Calculate and print the mean squared error (MSE) for the full model on the test set
print("The test MSE of the model is", round(mean_squared_error(y_test, y_pred), 2))
