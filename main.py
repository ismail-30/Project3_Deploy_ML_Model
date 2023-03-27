import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference



# Data Class Definition
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }


# Initialize FastAPI app
app = FastAPI()

# Create a global variable to store the models
MODELS = {}

# Load models


@app.on_event("startup")
async def startup_event():
    model_paths = {
        'model': 'model/model.pkl',
        'encoder': 'model/encoder.pkl',
        'lb': 'model/lb.pkl'}
    for path_key, path in model_paths.items():
        with open(path, 'rb') as file:
            MODELS[path_key] = pickle.load(file)
    return MODELS

# Show a welcome msg


@app.get('/')
async def welcome():
    return {'msg': 'Welcome to Census data salary prediction!'}

# Model Inference


@app.post('/predict')
async def predict(input: CensusData):
    """ Send POST request with input data
    Inputs
    ------
    input : CensusData
        Input Data
    models : dict
        Trained models for preprocessing and inference.
    Returns
    -------
     : dict
        Model Inference on Input Data
    """

    global MODELS  # Declare the global variable

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

    # Prepare Data
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])

    # Preprocess data
    X, _, _, _ = process_data(input_df, categorical_features=cat_features,
                              label=None, encoder=MODELS['encoder'], lb=MODELS['lb'], training=False)

    # Prediction
    preds = inference(MODELS['model'], X)

    return {'result': int(preds[0])}

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)
