# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The optimization algorithm is gradient boosting on decision trees, implemented by the [CatBoost](https://catboost.ai/)
open source library (Yandex). The model is trained and validated with k-fold cross-validation for a given number of
iterations, then the model selected at the iteration with the lowest validation
loss is re-trained on the whole validation dataset (all k folds).

## Intended Use

The model is meant for learning, to practice the development of a reproducible machine learning pipeline.
It predicts whether an individual living in the USA makes more than 50,000 USD a year.

## Training and Validation Data

The dataset has been adapted by Udacity from the
[UCI Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income). It contains 32561 samples overall.
The size of the training and validation set is 80% of the overall dataset, and has been used for hyper-parameters
tuning and model selection via k-fold cross-validation.

Splitting of the training and evaluation set in folds was stratified based on the variable to be predicted, to
ensure equal representation of the two classes to be predicted in every fold.

Pre-processing cleaned the formatting to make it more easily ingested by the CatBoost library; it also encoded the two
categories to be predicted with an integer: 1 for salary > 50 KUSD and 0 for salary <= 50 KUSD. CatBoost encodes
categorical variables automatically, therefore no encoding was done during pre-processing.

## Test Data

20% of the overall dataset has been held out of training and validation; it has been used to test the model and
measure its performance metrics.

The overall dataset has been split between training/validation set and test set in a stratified manner, based on the
variable to be predicted.

Pre-processing of the test set was the same as for the training/validation set.

## Metrics

_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations
Information in the dataset is dated, as it was compiled starting from the 1994 Census.