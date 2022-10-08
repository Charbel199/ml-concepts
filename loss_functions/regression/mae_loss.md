# Mean Absolute Error Loss

All errors in MAE are simply weighted on the same linear scale.

Here is the loss function equation:

![mse](../../docs/MAELoss.png)

Unlike the MSE, MAE doesn't put too much weight on  outliers and the loss function provides a generic and even measure of how well the model is performing.

## Reference(s)
[MAE Loss](https://torchmetrics.readthedocs.io/en/stable/regression/mean_absolute_error.html)