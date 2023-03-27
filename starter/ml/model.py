from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the hyperparameters to tune
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "random_state": [42]
    }

    # Create a Random Forest Classifier
    model = RandomForestClassifier()

    # Create a grid search object and fit it to the training data
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best estimator and return it
    best_estimator = grid_search.best_estimator_
    return best_estimator


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def performance_overall(model, X, y):
    """ Evalue performance on test data.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for testing.
    y: np.array
        test labels, binarized
    Returns
    -------
    metrics_dict :
        dictionary storing different metrics
        precision : float
        recall : float
        fbeta : float

    """

    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    return precision, recall, fbeta


def performance_on_slice(
        data,
        cat_features,
        trained_model,
        encoder,
        lb,
        feature):
    """ Function for data slicing model performance
       for a certain categorical column
    """
    # get distinct column category value
    unique_vals = data[feature].unique()

    metrics_dict = {}

    # iterate each value and record the metrics
    for val in unique_vals:
        # Fix the feature
        idx = data[feature] == val
        reduced_data = data[idx]

        # Process this subset of data for testing
        X, y, encoder, lb = process_data(
            reduced_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Do the inference and Compute the metrics
        preds = inference(trained_model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        metrics_dict[val] = {"precision": precision,
                             "recall": recall,
                             "fbeta": fbeta}

    return metrics_dict
