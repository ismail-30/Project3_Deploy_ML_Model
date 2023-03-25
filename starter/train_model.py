# Script to train machine learning model.

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add code to load in the data.
data = pd.read_csv("data/census_cleaned.csv")

# Transform target variable to 1 for salaries above 50k and 0 for salaries below or equal to 50k.
data["salary"] = data["salary"].apply(lambda x: int(x == ">50K"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
X_train, X_test, y_train, y_test = train_test_split(data.drop("salary", axis=1), data["salary"], test_size=0.20)

# Process the categorical data with label encoding.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

encoder = LabelEncoder()
for feature in cat_features:
    X_train[feature] = encoder.fit_transform(X_train[feature])
    X_test[feature] = encoder.transform(X_test[feature])

# Scale the numerical data.
num_features = ["age", "fnlgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Perform hyperparameter tuning on the random forest classifier.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
}
rf = RandomForestClassifier(random_state=42)
clf = GridSearchCV(rf, param_grid, cv=5)
clf.fit(X_train, y_train)

# Evaluate the model's accuracy using cross-validation.
scores = cross_val_score(clf.best_estimator_, X_test, y_test, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Save the model for future use.
joblib.dump(clf.best_estimator_, "model/rf_classifier.joblib")



