# Script to train machine learning model.
from ml.data import process_data
import ml.model
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

def run_ML_pipeline():

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
        test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
    )

    print("model training started ...")
    # Train and save a model.
    model = ml.model.train_model(X_train, y_train)
    #with open("model/model.pkl", "rb") as f:
    #    model = pickle.load(f)
    print("model training finished...")

    model_paths = {'model': 'model/model.pkl', 'encoder': 'model/encoder.pkl', 'lb': 'model/lb.pkl'}
    for path_key,path in model_paths.items():
        with open(path,'wb') as f:
            pickle.dump(eval(path_key),f)
    
    return model, X_test, y_test

def get_metrics_performance(model, X_test, y_test, feature_name):
    """
    Computes performance metrics for model slices based on the given feature.

    Inputs
    ------
    model :
    X_test:
    y_test:
    feature_name : Trained machine learning model.

    Outputs
    -------
    None. Metrics will be saved in seperate txt files.
    """
    feature_idx = column_name_to_index(feature_name)

    # Get metrics performances and save to output text files
    output_paths = {'overall': 'model/output.txt', 'slice': 'model/slice_output.txt'}
    for output_key, output_path in output_paths.items():
        # Slice data performances
        if output_key == 'slice':
            metrics = ml.model.performance_on_slice(model, X_test, y_test, feature_idx)
        # Overall data performances
        else:
            metrics = ml.model.performance_overall(model, X_test, y_test)

        # Save to output text
        with open(output_path,'w') as f:
            if output_key == 'slice':
                for value, metric in metrics.items():
                    f.write(f"Metrics for {feature_name}:\n")
                    for metric_name, metric_value in metric.items():
                        f.write(f"{metric_name}: {metric_value:.2f}\n")
                    f.write("\n")
            else:
                f.write("precision,recall,fbeta\n")
                f.write(",".join(map(str, metrics)))

def column_name_to_index(column_name):
    """
    finds the index of a given column name in Census data file.

    Inputs
    ------
    column_name :
    
    Outputs
    -------
    column_index. 
    """
    data = pd.read_csv('data/census_cleaned.csv')
    try:
        column_index = data.columns.get_loc(column_name)
        return column_index
    except KeyError:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None
        

if __name__ == '__main__':
    model, X_test, y_test = run_ML_pipeline()
    get_metrics_performance(model, X_test, y_test, "education")