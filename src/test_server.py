from fastapi.testclient import TestClient
import pytest
from pathlib import Path
from server import app, Sample
from train import eval_model

raw_datafile = '../data/census.csv'
cleaned_datafile = '../data/census_cleaned.csv'
saved_model = '../model/trained_model.bin'
categorical_idx = [1, 3, 4, 5, 6, 7, 8, 9, 13]


@pytest.fixture(scope='session')
def setup():
    assert Path(saved_model).exists(), f'Model must be saved in f{saved_model} in order to test the API.'
    client = TestClient(app)
    return client


def test_home(setup):
    client = setup
    res = client.get('/')
    assert res.status_code == 200


def test_perform_inference_neg(setup):
    client = setup
    sample_dict = {"age": 39,
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
                   "native_country": "United-States"}
    res = client.post('/inference/',
                      json=sample_dict)
    assert res.status_code == 200
    pred_prob = res.json()
    assert len(pred_prob) == 1
    assert pred_prob.get('predicted_probability >50K',
                         None) is not None, 'The API call should return JSON data with one dictionary with only one key in it.'
    assert pred_prob['predicted_probability >50K'] < .5, 'The sample should have been classified as 0 (negative)'


def test_perform_inference_pos(setup):
    client = setup
    sample_dict = {"age": 52,
                   "workclass": "Self-emp-not-inc",
                   "fnlgt": 209642,
                   "education": "HS-grad",
                   "education_num": "9",
                   "marital_status": "Married-civ-spouse",
                   "occupation": "Exec-managerial",
                   "relationship": "Husband",
                   "race": "White",
                   "sex": "Male",
                   "capital_gain": 0,
                   "capital_loss": 0,
                   "hours_per_week": 45,
                   "native_country": "United-States"}
    res = client.post('/inference/',
                      json=sample_dict)
    assert res.status_code == 200
    pred_prob = res.json()
    assert len(pred_prob) == 1
    assert pred_prob.get('predicted_probability >50K',
                         None) is not None, 'The API call should return JSON data with one dictionary with only one key in it.'
    assert pred_prob['predicted_probability >50K'] > .5, 'The sample should have been classified as 1 (positive)'
