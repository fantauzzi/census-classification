import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
# from sklearn.metrics import log_loss
from catboost import Pool, cv, CatBoostClassifier
from census_preprocess import pre_process, str_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

seed = 42
raw_datafile = '../data/census.csv'
cleaned_datafile = '../data/census_cleaned.csv'
saved_model = '../model/trained_model.bin'
categorical_idx = [1, 3, 4, 5, 6, 7, 8, 9, 13]


def train_and_save(train_val_pool: Pool, filename: str, file_format: str = 'cbm') -> CatBoostClassifier:
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
    params = {}
    params['loss_function'] = 'Logloss'
    params['iterations'] = 50
    # params['custom_loss'] = 'AUC'
    params['random_seed'] = seed
    params['learning_rate'] = 0.5

    logging.info(f'Training with cross-validation')
    cv_data = cv(params=params,
                 pool=train_val_pool,
                 fold_count=5,
                 partition_random_seed=seed,
                 stratified=True,
                 verbose=True)
    best_iter = np.argmin(cv_data['test-Logloss-mean'])
    best_loss = cv_data.loc[best_iter, 'test-Logloss-mean']
    logging.info(f'Best iteration among all folds was #{best_iter} with test loss {best_loss}.')
    logging.info(f'Re-training the model for {best_iter + 1} iterations on the whole training/validation dataset')

    params['iterations'] = best_iter + 1
    best_model = CatBoostClassifier(verbose=True,
                                    **params)
    best_model.fit(train_val_pool,
                   verbose=True)

    best_model.save_model(fname=filename,
                          format=file_format,
                          pool=train_val_pool if format in ('cpp', 'python', 'JSON') else None)
    return best_model


def test_model(file_name: str, test_pool: Pool, y_test: list, metrics: list[str]) -> dict[str,float]:
    logging.info(
        f'Evaluating model saved in {file_name} against a test set with {len(y_test)} samples ({sum(y_test)} positive).')
    model = CatBoostClassifier()
    model.load_model(file_name)
    y_prob = model.predict_proba(test_pool)
    # test_loss = log_loss(y_test, y_prob)
    metrics = model.eval_metrics(data=test_pool,
                                 metrics=metrics,
                                 ntree_start=model.tree_count_ - 1)
    # Change the values in the metrics dict, so they are scalars instead of lists of length 1
    metrics = {key: value[0] for key, value in metrics.items()}
    logging.info(f'Evaluation results:\n{metrics}')
    return metrics


def main():
    if not Path(cleaned_datafile).exists():
        logging.info(f'Pre-processed dataset {cleaned_datafile} not found. Going to make it now from raw dataset {raw_datafile}')
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
        f'Got {len(X_train)} samples in the training set ({sum(y_train)} positive) and {len(X_test)} in the test set ({sum(y_test)} positive).')

    # Make a Pool for CatBoost out of the training set
    train_val_pool = Pool(data=X_train, label=y_train, cat_features=categorical_idx)

    # Train and cross-validate the model, and save it in CatBoost binary format
    model = train_and_save(train_val_pool=train_val_pool,
                           filename=saved_model,
                           file_format='cbm')

    # Test the model
    test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_idx)
    metrics = test_model(saved_model, test_pool, y_test, ['Logloss', 'AUC', 'F1', 'Recall', 'Precision', 'Accuracy'])


if __name__ == '__main__':
    main()
