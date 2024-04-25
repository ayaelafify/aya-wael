

from ucimlrepo import fetch_ucirepo 

# Fetch dataset
breast_cancer = fetch_ucirepo(id=14) 

# Data (as pandas dataframes)
X = breast_cancer.data.features 
y = breast_cancer.data.targets 

# Name of the columns
column_names = breast_cancer.variables

# Number of rows and columns
num_rows = X.shape[0]
num_columns = X.shape[1]

# Display first 10 rows
first_10_rows = X[:10]

# Display last 10 rows
last_10_rows = X[-10:]

# Print the results
print("Column names:", column_names)
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)
print("First 10 rows:\n", first_10_rows)
print("Last 10 rows:\n", last_10_rows)

#next we will convert the dataset to a pandas DataFrame, 
# calculate the basic statistics using the describe() function
#  then  check for missing values using the isnull().sum() function
import pandas as pd

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(X, columns=column_names)

# Basic statistics
statistics = df.describe()

# Missing values
missing_values = df.isnull().sum()

# Print the results
print("Basic statistics:\n", statistics)
print("Missing values:\n", missing_values)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode categorical features (if any)
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_imputed)


#uses the train_test_split function from scikit-learn to split the preprocessed data into training and testing sets.
#  It assigns 80% of the data to the training set (X_train and y_train) and 20% of the data to the testing set (X_test and y_test).


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Choose the classification/regression algorithm
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the testing data
y_pred = model.predict(X_test)
#uses the accuracy_s
# Calculate classification/regression metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


#we used the accuracy_score, precision_score, recall_score, and f1_score functions from scikit-learn to calculate the classification metrics.
#  The evaluation results are then printed
# Print the evaluation results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# we use  the LogisticRegression class from scikit-learn as an example of  a classification algorithm. 
# The fit method is then used to train the model on the training data.
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Train the best model on the training data
best_model.fit(X_train, y_train)