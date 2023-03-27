"""
    Script to train machine learning model.
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import ml.model
from ml.data import process_data



def run_ml_pipeline(feature):

    # Add code to load in the data.
    data = pd.read_csv('data/census_cleaned.csv')

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", encoder=encoder,
        lb=lb, training=False)

    print("model training started ...")
    # Train and save a model.
    model = ml.model.train_model(X_train, y_train)
    print("model training finished...")

    model_paths = {
        'model': 'model/model.pkl',
        'encoder': 'model/encoder.pkl',
        'lb': 'model/lb.pkl'}
    for path_key, path in model_paths.items():
        with open(path, 'wb') as file:
            pickle.dump(eval(path_key), file)

    # Get metrics performances and save to output text files
    output_paths = {
        'overall': 'model/output.txt',
        'slice': 'model/slice_output.txt'}
    for output_key, output_path in output_paths.items():
        # Slice data performances
        if output_key == 'slice':
            metrics = ml.model.performance_on_slice(
                test, cat_features, model, encoder, lb, feature)
        # Overall data performances
        else:
            metrics = ml.model.performance_overall(model, X_test, y_test)

        # Save to output text
        with open(output_path, 'w') as file:
            if output_key == 'slice':
                for value, metric in metrics.items():
                    file.write(f"Metrics for {feature}: {value}\n")
                    for metric_name, metric_value in metric.items():
                        file.write(f"{metric_name}: {metric_value:.2f}\n")
                    file.write("\n")
            else:
                file.write("precision,recall,fbeta\n")
                file.write(",".join(map(str, metrics)))


if __name__ == '__main__':
    run_ml_pipeline("education")
