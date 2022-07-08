# Early Stopping

Early stopping consists of running the training process without specifying the number of iterations/epoch. The early stopping algorithm
will automatically detect when the loss has stabilized and stop the training.

![early](../assets/EarlyStopping.png)

The downside of early stopping is that it might affect other aspects of the model, other than overfitting. L2 regularization
is usually the better solution for high variance problems.

## Reference(s)
[Early Stopping](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)