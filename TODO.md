# TODO

## EDA
 - Load data
 - Do pandas-profiling
 - Check outliers, missing data, issues with values formatting (extra spaces?)
 - Check the categories balance (how relevant is it for gradient boosting?)
 - If necessary, do data cleaning and then repeat the EDA on the cleaned data

## Data cleaning
 - find out from the EDA what needs to be done here
 - Perhaps encode the features to be predicted (classification)

## Training
- Set the categorical variables, for Catboost to encode
- Decide what kind of cross-validation, hold-out or k-fold
- Set hyper-parameters
- Train and validate

## Tuning and testing
- Optimize hyper-parameters re-training the model as needed
- Save each hyper-parameters value combination as they are tried, along with the corresponding metrics value
- Chose the best combo of hyper-parameters. Retrain the model on train+val dataset and save the trained model for inference
- Load the saved model and test it 


## Inference
- Load the model
- Do the inference as directed by command line (batch only?) or API (single sample only?)

