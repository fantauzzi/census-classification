import pytest
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from catboost import CatBoostClassifier, Pool
from pathlib import Path
from pandas import read_csv, notnull
from numpy import ndarray
from census_preprocess import str_columns, pre_process
from train import train_and_save, predict_proba, eval_model

if not GlobalHydra().is_initialized():
    hydra.initialize_config_dir(config_dir=os.getcwd(), version_base='1.1')
params = hydra.compose(config_name='params.yaml')
raw_datafile = params['raw_data']
test_cleaned_datafile = params['unit-testing']['test_cleaned_data']
test_saved_model = params['unit-testing']['test_save_model']
categorical_idx = params['model']['categorical_idx']
classifier_params = dict(params['catboost_classifier'])
cv_params = dict(params['catboost_cv'])


@pytest.fixture(scope='session')
def setup():
    """
    Set-up to be performed only once after any of the tests in this file.
    :return: a pair with the whole dataset, except the variable to be predicted, and the variable to be predicted
    :rtype: pandas.DataFrame, pandas.DataFrame
    """
    Path(test_cleaned_datafile).unlink(missing_ok=True)
    pre_process(raw_datafile, test_cleaned_datafile)
    assert Path(test_cleaned_datafile).exists()
    df = read_csv(test_cleaned_datafile)

    str_column_names = str_columns(df)
    # Replace any null value in columns of type string with the string 'None', in order to be processed correctly
    # by CatBoost

    for col_name in str_column_names:
        df[col_name] = df[col_name].where(notnull(df[col_name]), 'None')

    # Train on the whole dataset: it is for testing purpose, no need to split a test set.
    X = df.drop('salary', axis=1)
    y = df['salary']
    yield X, y

    # Teardown after all testing is done
    Path(test_cleaned_datafile).unlink(missing_ok=True)
    Path(test_saved_model).unlink(missing_ok=True)


def test_train_and_save(setup):
    X, y = setup
    pool = Pool(data=X, label=y, cat_features=categorical_idx)
    model = train_and_save(train_val_pool=pool,
                           filename=test_saved_model,
                           file_format='cbm',
                           classifier_params=classifier_params,
                           cv_params=cv_params)
    assert type(model) == CatBoostClassifier, 'Trained model must be of type CatBoostClassifier'
    assert Path(test_saved_model).exists(), 'After training the trained model should be written in a file'


def test_predict_proba(setup):
    X, _ = setup
    res = predict_proba(test_saved_model, X)
    assert len(res) == len(
        X), 'The number of items in the inference should be the same as the number of samples in the dataset'
    assert type(res) == ndarray, 'The inference should be an numpy array'
    assert (res >= 0).all(), 'Inferences should be probabilities, between 0 and 1'
    assert (res <= 1).all(), 'Inferences should be probabilities, between 0 and 1'


def test_eval_model(setup):
    X, y = setup
    metrics_name = ['Logloss', 'AUC', 'F1', 'Recall', 'Precision', 'Accuracy']
    pool = Pool(data=X, label=y, cat_features=categorical_idx)
    if not Path(test_saved_model).exists():
        train_and_save(train_val_pool=pool,
                       filename=test_saved_model,
                       classifier_params=classifier_params,
                       cv_params=cv_params,
                       file_format='cbm')
    metrics_value = eval_model(test_saved_model, pool, metrics_name)
    assert type(metrics_value) == dict, 'Metrics should be returned in a dictionary'
    keys = metrics_value.keys()
    assert {*keys} == {*metrics_name}, 'Returned metrics should be the requested CatBoost metrics'
