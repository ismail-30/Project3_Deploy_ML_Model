import requests

api_endpoint = "https://udacity-project3.herokuapp.com/"

payload = {
    "age": 27,
    "workclass": "State-gov",
    "fnlgt": 94253,
    "education": "9-th",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 1945,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

response = requests.post(api_endpoint + "predict", json=payload)


print(f"Subject:{payload}")
print(f'Status Code:{response.status_code}')
print(f"Prediction:{response.content}")
