# Import necessary libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 20)
plt.rcParams["figure.figsize"] = (12,8)
import pandas as pd

# Read the training data from a CSV file
elect_train = pd.read_csv("data/county_election_train.csv")

# Read the test data from a CSV file
elect_test = pd.read_csv("data/county_election_test.csv")

# Show the first few rows of the training data
print(elect_train.head())

# Make a new variable that says who got more votes in each row (1 if Trump, 0 if Clinton)

# For training data: 1 if Trump got more votes than Clinton, else 0
y_train = (elect_train["trump"] > elect_train["clinton"]).astype(int)

# For test data: 1 if Trump got more votes than Clinton, else 0
y_test = (elect_test["trump"] > elect_test["clinton"]).astype(int)
import matplotlib.pyplot as plt

# Make a scatter plot to compare "minority" and "bachelor" values in the training data

# Show Trump counties in blue
plt.scatter(
    elect_train.loc[y_train == 1, "minority"], 
    elect_train.loc[y_train == 1, "bachelor"],
    marker=".",
    color="blue",
    label="Trump",
    s=50,
    alpha=0.4
)

# Show Clinton counties in green
plt.scatter(
    elect_train.loc[y_train == 0, "minority"], 
    elect_train.loc[y_train == 0, "bachelor"],
    marker=".",
    color="green",
    label="Clinton",
    s=50,
    alpha=0.4
)

plt.xlabel("minority")      # Label for x-axis
plt.ylabel("bachelor")     # Label for y-axis
plt.legend()               # Add a legend to explain colors
plt.show()                 # Show the plot

from sklearn.tree import DecisionTreeClassifier

# Make a decision tree model (depth 3, uses Gini for splitting)
dtree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)

# Train the model using only the "minority" column to predict Trump vs Clinton
dtree.fit(elect_train[['minority']], y_train)

# Make the plot big
plt.figure(figsize=(30, 20))

# Show the decision tree as a picture
# Shows how the model splits "minority" to guess Clinton or Trump
tree.plot_tree(
    dtree, 
    feature_names=['minority'], 
    class_names=['Clinton', 'Trump'], 
    filled=True
)

# Show the plot on the screen
plt.show()
