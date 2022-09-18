# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The optimization algorithm is gradient boosting on decision trees, implemented by the [CatBoost](https://catboost.ai/)
open source library (Yandex). The model is trained and validated with k-fold cross-validation for a given number of
iterations, then the model selected at the iteration with the lowest validation
loss is re-trained on the whole validation dataset (all k folds).

## Intended Use

The model is meant for learning, to practice the development of a reproducible machine learning pipeline.

It predicts whether an individual living in the USA makes more than 50,000 USD a year. The prediction is expressed
in terms of probability, as a number between 0 and 1, that the individual makes more than 50,000 USD.

## Training and Validation Data

The dataset has been adapted by Udacity from the
[UCI Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income). It contains 32561 samples overall.
The size of the training and validation set is 80% of the overall dataset, and has been used for hyper-parameters
tuning and model selection via k-fold cross-validation.

Splitting of the training and evaluation set in folds was stratified based on the variable to be predicted, to
ensure equal representation of the two classes to be predicted in every fold.

Pre-processing cleaned the data format to make it more easily ingested by the CatBoost library; it also encoded the two
categories to be predicted with an integer: 1 for salary > 50 KUSD and 0 for salary <= 50 KUSD. CatBoost encodes
categorical variables automatically, therefore no encoding was done during pre-processing.

## Test Data

20% of the overall dataset has been held out of training and validation; it has been used to test the model and
measure its performance metrics.

The overall dataset has been split between training/validation set and test set in a stratified manner, based on the
variable to be predicted.

Pre-processing of the test set was the same as for the training/validation set.

## Metrics

With 1000 iterations of gradient boosting, a 5-fold cross-validation and a lerning rate of 0.03, 
these are the performance metrics of the trained model on the test set:

 - Logloss 0.2726851302896537
 - AUC 0.9318919466168671
 - F1 0.7181628392484342
 - Recall 0.6581632653061225
 - Precision 0.7901990811638591
 - Accuracy 0.8756333486872409

Training and cross-validation took less than one minute on an Intel i7-10700K CPU 
with an NVIDIA GTX-1080 GPU.

## Ethical Considerations
The variable to be predicted is binary, indicating whether a person makes more than 50 KUSD/year. The 50 KUSD threshold
corresponds to a specific quantile for different populations, e.g. the 76th quantile for all individuals, but the 89th 
quantile among women. Consequently, findings related to algorithmic fairness are sensitive to the choice of income
threshold. Ref. [Ding, Frances & Hardt, Moritz & Miller, John & Schmidt, Ludwig. (2021). Retiring Adult: New Datasets 
for Fair Machine Learning](https://openreview.net/pdf?id=bYi_2708mKK). 

## Caveats and Recommendations
Information in the dataset is dated, as it was compiled starting from the 1994 Census. Metrics collected on slices
of the dataset show that for some values of the categorical variables (e.g. for variable "education") there are no 
positive or no negative samples, and anyway a modest sample size. A more recent and larger dataset should be extracted
from the Census Dataset for any purpose that is not didactic.