import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import Pool, cv, CatBoostClassifier
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

seed = 42
cleaned_datafile = '../data/census_cleaned.csv'
categorical_idx = [1, 3, 4, 5, 6, 7, 8, 9, 13]
df = pd.read_csv(cleaned_datafile)
# print(df.iloc[:, categorical_idx])

str_column_names = [name for name in df.columns if pd.api.types.is_string_dtype(df[name])]
# Remove any leading/trailing blank from strings in the dataframe, and replace unknwon values ('?') with a None
for col_name in str_column_names:
    df[col_name] = df[col_name].where(pd.notnull(df[col_name]), 'None')
X = df.drop('salary', axis=1)
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=seed, stratify=y)
logging.info(
    f'Got {len(X_train)} samples in the training set ({sum(y_train)} positive) and {len(X_test)} in the test set ({sum(y_test)} positive).')

train_val_pool = Pool(data=X_train, label=y_train, cat_features=categorical_idx)
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
model = CatBoostClassifier(verbose=True,
                           **params)
model.fit(train_val_pool,
          verbose=True)

logging.info(f'Evaluating the model against the test set.')
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_idx)
y_prob = model.predict_proba(test_pool)
# test_loss = model.eval_metrics(test_pool, params['loss_function'])
test_loss = log_loss(y_test, y_prob)
logging.info(f'Test loss is {test_loss}')

