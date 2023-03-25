import joblib

model = joblib.load("model/rf_classifier.joblib")

# Define the categorical features to slice by.
categorical_features = ["race", "gender"]

for feature in categorical_features:
    unique_values = test_data[feature].unique()
    for value in unique_values:
        print(value)

