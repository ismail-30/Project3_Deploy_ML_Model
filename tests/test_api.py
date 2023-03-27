from fastapi.testclient import TestClient
from main import app
import pytest

@pytest.fixture
def client():
    with TestClient(app) as cli:
        yield cli

@pytest.fixture
def payload_over_50K_sample():
    return {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14000,
        "capital_loss": 0,
        "hours_per_week": 55,
        "native_country": "United-States",
    }

@pytest.fixture
def payload_under_50K_sample():
    return {
        "age": 22,
        "workclass": "Private",
        "fnlgt": 32696,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30,
        "native_country": "United-States",
    }

@pytest.fixture
def payload_error_sample():
    return {
        "age": "twenty",  # Invalid data type
    }

# GET: Welcome message
def test_welcome(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'msg': 'Welcome to Census data salary prediction!'}

# POST: payload with over 50K input
def test_predict_over_50K(client, payload_over_50K_sample):
    response = client.post('/predict', json=payload_over_50K_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 1

# POST: payload with under 50K input
def test_predict_under_50K(client, payload_under_50K_sample):
    response = client.post('/predict', json=payload_under_50K_sample)
    assert response.status_code == 200
    assert response.json()['result'] == 0

# POST: Errorneous input
def test_predict_error(client, payload_error_sample):
    response = client.post('/predict', json=payload_error_sample)
    assert response.status_code != 200