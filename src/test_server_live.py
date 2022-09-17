import requests
import os
import hydra
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra().is_initialized():
    hydra.initialize_config_dir(config_dir=os.getcwd(), version_base='1.1')

params = hydra.compose(config_name='params.yaml')
url = params['unit-testing']['live_url']


def test_home():
    res = requests.get(url)
    assert res.status_code == 200
    message = res.json()
    assert type(message) == dict
    assert len(message) == 1
    assert message['message'] == 'Inference for census classification'


def test_inference():
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
    res = requests.post(f'{url}inference/',
                        json=sample_dict)
    assert res.status_code == 200
    print(f'\nStatus code from POST request to live API on {url} is {res.status_code}')
    pred_prob = res.json()
    assert len(pred_prob) == 1
    assert pred_prob.get('predicted_probability >50K',
                         None) is not None, 'The API call should return JSON data with one dictionary with only one key in it.'
    assert pred_prob['predicted_probability >50K'] < .5, 'The sample should have been classified as 0 (negative)'
    print(f"The model inference (prob. of class '>50K') is {pred_prob['predicted_probability >50K']}")
