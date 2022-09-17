from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from train import predict_proba
import os
import hydra
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra().is_initialized():
    hydra.initialize_config_dir(config_dir=os.getcwd(), version_base='1.1')
params = hydra.compose(config_name='params.yaml')
app = FastAPI()


class Sample(BaseModel):
    """
    A BaseModel to parse one sample to be used for inference. Note that some field names in the dataset
    have hyphens (e.g. `capital-gain`) but Python syntax doesn't allow them here, therefore replacing them with
    underscores (e.g. `capital_gain`).
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get('/')
async def home() -> dict[str, str]:
    """
    The root of the API.
    :return: A greeting message.
    :rtype: dict[str, str]
    """
    return {'message': 'Inference for census classification'}


# age	workclass	fnlgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country	salary

@app.post('/inference/')
async def perform_inference(sample: Sample) -> dict[str, float]:
    """
    Performs an inference with the trained model for a given sample and returns the result.
    :param sample: The given sample.
    :type sample: Sample
    :return: {'predicted_probability >50K': y_pred} where `y_pred` is the probability that the given samples belongs
    to the classe >50K, a real number between 0 and 1.
    :rtype: dict[str, float]
    """
    saved_model = params['save_model']
    sample = dict(sample)
    # Some field names defined in the BaseModel have underscores, but the corresponding column names (in the dataset)
    # have hyphens instead, therefore replace underscores with hyphens in the column names.
    sample = {key.replace('_', '-') if type(key) == str else key: value for key, value in sample.items()}
    sample_df = pd.DataFrame(sample, index=[0])
    y_pred = predict_proba(saved_model, sample_df)[0][1]
    result = {'predicted_probability >50K': y_pred}
    return result


"""
One sample POST request, kept here for convenience:

curl -X 'POST' \
  'http://127.0.0.1:8000/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": "13",
  "marital_status": "marital-status",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}'
"""
