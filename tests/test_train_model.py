import numpy as np
import pandas as pd
import pickle
import pytest
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, performance_on_slice, performance_overall

# Load pre-trained model
with open("model/model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


@pytest.fixture
def raw_data():
    # Load raw data
    data = pd.read_csv('data/census_cleaned.csv')
    return data


@pytest.fixture
def split_data(raw_data):
    # Split data into train and test sets
    train, test = train_test_split(raw_data, test_size=0.20, random_state=42)
    return train, test


@pytest.fixture
def preprocessed_data(split_data):
    train, _ = split_data
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
    return X_train, y_train, encoder, lb


@pytest.fixture
def trained_model(preprocessed_data):
    _, _, encoder, lb = preprocessed_data
    return loaded_model, encoder, lb


def test_preprocess_data(preprocessed_data):
    X_train, y_train, encoder, lb = preprocessed_data
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert encoder is not None
    assert lb is not None


def test_compute_model_metrics(trained_model, preprocessed_data):
    model, encoder, lb = trained_model
    X_train, y_train, _, _ = preprocessed_data
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision >= 0.7
    assert recall >= 0.5
    assert fbeta >= 0.6


def test_performance_on_slice(trained_model, preprocessed_data):
    model, encoder, lb = trained_model
    X_train, y_train, _, _ = preprocessed_data
    feature_idx = 1  # education feature index
    metrics = performance_on_slice(model, X_train, y_train, feature_idx)
    for value, metric in metrics.items():
        for metric_name, metric_value in metric.items():
            assert isinstance(metric_value, float)


def test_performance_overall(trained_model, preprocessed_data):
    model, encoder, lb = trained_model
    X_train, y_train, _, _ = preprocessed_data
    metrics = performance_overall(model, X_train, y_train)
    assert len(metrics) == 3