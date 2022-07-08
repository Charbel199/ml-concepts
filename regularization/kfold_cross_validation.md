# K-fold Cross Validation

In general cases, machine learning datasets are usually split between training and test sets.
These sets are in turn respectively used for training the model and testing the final model.
<br/>

Getting more into details:
* The training dataset is usually split into: Training and validation datasets.
  * Training Dataset: Used to fit the model
  * Validation Dataset: Used to evaluate model fit during training while tuning hyperparameters.
<br/>
  
For a better and unbiased estimate of the model skill while training, the **K-fold Cross Validation** approach is used where
several identical models are trained with the only difference in each iteration being the validation dataset:
<br/>
<br/>
Here is an illustration showing the concept of K-fold Cross Validation:
<br/>

![kfold](../assets/KFoldCrossValidaiton.png)

<br/>

Benefits of K-fold Cross Validation:

* Ensures that every data point has a change of appearing in the training and validation datasets
* Helps in detecting overfitting
* Average every fold's score for a more general score estimate of the model

## Reference(s)
[Sci-kit learn cross  validation](https://scikit-learn.org/stable/modules/cross_validation.html)