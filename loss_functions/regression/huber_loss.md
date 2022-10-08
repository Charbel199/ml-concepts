# Huber Loss

The huber loss is the middle-ground between MAE and MSE.

Here is the loss function equation:

![mse](../../docs/HuberLoss.png)

It does take into consideration the necessity of increasing the significance of outlier errors.
For all values that are less than a value 'delta', it will use the MSE loss and for all values greater than delta,
it will use the MAE loss.

## Reference(s)
[Huber Loss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)