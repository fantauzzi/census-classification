from os import getcwd
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import Pool, cv, CatBoostClassifier
import hydra
from omegaconf import DictConfig
from census_preprocess import pre_process, str_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train_and_save(train_val_pool: Pool,
                   filename: str,
                   file_format: str,
                   classifier_params: dict,
                   cv_params: dict) -> CatBoostClassifier:
    """
    Train and select a classification model based on cross-validation, save and return it. Before saving, the
    selected model is re-trained on the whole training and validation dataset.

    :param train_val_pool: The CatBoost Pool holding the dataset for training and cross-validation.
    :type train_val_pool: catboost.Pool
    :param filename: The name of the file to be used to save the trained model.
    :type filename: str
    :param file_format: The type for the model save file, see CatBoost documentation for
    [save_model](https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model).
    :type file_format: str
    :return: The trained classification model, the same that went into the save file.
    :rtype: catboost.CatBoostClassifier
    """

    logging.info('Training with cross-validation.')
    classifier_params['loss_function'] = 'Logloss'
    cv_data = cv(params=classifier_params,
                 pool=train_val_pool,
                 **cv_params)
    best_iter = np.argmin(cv_data['test-Logloss-mean'])
    best_loss = cv_data.loc[best_iter, 'test-Logloss-mean']
    logging.info(f'Best iteration among all folds was #{best_iter} with test loss {best_loss}.')
    logging.info(f'Re-training the model for {best_iter + 1} iterations on the whole training/validation dataset')

    classifier_params['iterations'] = best_iter + 1
    best_model = CatBoostClassifier(**classifier_params)  # verbose=True,
    best_model.fit(train_val_pool)
    # verbose=True)

    best_model.save_model(fname=filename,
                          format=file_format,
                          pool=train_val_pool if format in ('cpp', 'python', 'JSON') else None)
    return best_model


def eval_model(file_name: str, test_pool: Pool, metrics: list[str]) -> dict[str, float]:
    logging.info(f'Evaluating model saved in {file_name}')
    model = CatBoostClassifier()
    model.load_model(file_name)
    metrics = model.eval_metrics(data=test_pool,
                                 metrics=metrics,
                                 ntree_start=model.tree_count_ - 1)
    # Change the values in the metrics dict, so they are scalars instead of lists of length 1
    metrics = {key: value[0] for key, value in metrics.items()}
    logging.info(f'Evaluation results:\n{metrics}')
    return metrics


def validate_given_slice(model: CatBoostClassifier,
                         X_test: pd.DataFrame,
                         y_test: list,
                         cat_features: list[int],
                         var_name: str, category: str,
                         metrics: list[str]) -> dict[str, float]:
    X_slice = X_test[X_test[var_name] == category]
    y_slice = y_test[X_test[var_name] == category]
    test_pool = Pool(data=X_slice, label=y_slice, cat_features=cat_features)
    logging.info(
        f'Slice for category "{category}" contains {len(X_slice)} samples ({sum(y_slice)} positive)')
    slice_metrics = model.eval_metrics(data=test_pool,
                                       metrics=metrics,
                                       ntree_start=model.tree_count_ - 1)
    logging.info(f'   Metrics for the slice: {slice_metrics}')
    return slice_metrics


def validate_model_slice(file_name: str,
                         X_test: pd.DataFrame,
                         y_test: list,
                         cat_features: list[int],
                         metrics: list[str],
                         var_name: str) -> dict[str, dict[str, float]]:
    # Only implemented for categorical variables
    assert pd.api.types.is_string_dtype(X_test[var_name]), 'Variable for slice testing must be categorical (string)'

    model = CatBoostClassifier()
    model.load_model(file_name)

    # Collect all possible values for the given variable
    categories = pd.unique(X_test[var_name])
    logging.info(f'Performing model slice testing for variable {var_name} with {len(categories)} categories:')
    for category in categories:
        logging.info(f'   "{category}"')

    slices_metrics = {}
    for category in categories:
        slices_metrics[category] = validate_given_slice(model=model,
                                                        X_test=X_test,
                                                        y_test=y_test,
                                                        cat_features=cat_features,
                                                        var_name=var_name,
                                                        category=category,
                                                        metrics=metrics)

    return slices_metrics


def predict_proba(model_filename: str, df: pd.DataFrame) -> np.ndarray:
    logging.info(f'Working directory for predic_proba() is {getcwd()}')
    logging.info(
        f'Making prediction for batch of {len(df)} samples.')
    if predict_proba.model is None:
        logging.info(f'Loading trained model {model_filename}')
        predict_proba.model = CatBoostClassifier()
        predict_proba.model.load_model(model_filename)
    y_prob = predict_proba.model.predict_proba(df, verbose=True)
    return y_prob


predict_proba.model = None


@hydra.main(version_base=None, config_path=".", config_name="params.yaml")
def main(params: DictConfig) -> None:
    logging.info(f'Working directory is {getcwd()}')

    seed = 42
    raw_datafile = params['raw_data']
    cleaned_datafile = params['cleaned_data']
    saved_model = params['save_model']
    categorical_idx = params['model']['categorical_idx']
    classifier_params = params['catboost_classifier']
    cv_params = params['catboost_cv']

    if not Path(cleaned_datafile).exists():
        logging.info(
            f'Pre-processed dataset {cleaned_datafile} not found. Going to make it now from raw dataset {raw_datafile}')
        pre_process(raw_datafile, cleaned_datafile)

    logging.info(f'Loading pre-processed dataset {cleaned_datafile}')
    df = pd.read_csv(cleaned_datafile)

    str_column_names = str_columns(df)
    # Replace any null value in columns of type string with the string 'None', in order to be processed correctly
    # by CatBoost

    for col_name in str_column_names:
        df[col_name] = df[col_name].where(pd.notnull(df[col_name]), 'None')

    # Split the dataset into training and test set. The training set will be used for training and cross-validation.
    X = df.drop('salary', axis=1)
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=seed, stratify=y)
    logging.info(
        f'Got {len(X_train)} samples in the training set ({sum(y_train)} positive) and {len(X_test)} \
        in the test set ({sum(y_test)} positive).')

    # Make a Pool for CatBoost out of the training set
    train_val_pool = Pool(data=X_train, label=y_train, cat_features=categorical_idx)

    # Train and cross-validate the model, and save it in CatBoost binary format
    train_and_save(train_val_pool=train_val_pool,
                   filename=saved_model,
                   file_format='cbm',
                   classifier_params=dict(classifier_params),
                   cv_params=dict(cv_params))

    # Test the model
    metrics_name = ['Logloss', 'AUC', 'F1', 'Recall', 'Precision', 'Accuracy']
    test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_idx)
    eval_model(saved_model, test_pool, metrics_name)

    validate_model_slice(file_name=saved_model,
                         X_test=X_test,
                         y_test=y_test,
                         cat_features=categorical_idx,
                         metrics=metrics_name,
                         var_name='education')


if __name__ == '__main__':
    main()
