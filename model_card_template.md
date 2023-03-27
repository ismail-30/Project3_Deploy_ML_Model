# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A RandomForestClassifer with scikit-learn's default hyperparameters other than random_state(42) are used. Trained with sklearn version 0.24.1. For dataset, the 1994 US Census's US Census Income Data Set is used

## Intended Use
This model is to predict whether income salary is larger than $50,000.


## Training Data
With scikit-learn's train_test_split function, 80% of the dataset was used for training purposes.


## Evaluation Data
With scikit-learn's train_test_split function, 20% of the dataset was used for evaluation purposes.


## Metrics
To evaluate the model performance, three metrics were used, namely, precision, recall and F1 score. These performance metrics on the model were as follows.
precision -->
recall -->
Fbeta-score -->


## Ethical Considerations
This particular data, like other census information, was obtained through surveys conducted among individuals. However, there is uncertainty regarding the comprehensiveness of the census due to potential biases introduced by both the surveyors and the surveyed individuals themselves

## Caveats and Recommendations
Note that the data used is almost 30 years old. While some might still hold, it is recommended to get fresh up-to-date data

