# Regression Metrics

Here are the most relevant regression metrics:

## Mean Squared Error
MSE represents the squared distance between actual and predicted values.
The difference is squared to avoid the cancellation of positive values with negative values.
<br />
<br />
![MSE](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum%20%28y%20-%20%5Chat%7By%7D%29%5E%7B2%7D)
<br />
One thing to note is that MSE is not very robust to outliers.

## Root Mean Squared Error
RMSE is just the square root of the MSE.
<br />
<br />
![RMSE](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7BRMSE%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%20%5Csum%20%28y%20-%20%5Chat%7By%7D%29%5E%7B2%7D%7D)
<br />
By applying a square root, the metric will have the same unit as the required output,
this makes it easier for interpretation.
## Mean Absolute Error
Mean absolute error is a very simple performance metric, it is robust to outliers.
<br />
<br />
![MAE](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7BMAE%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum%20%5Cleft%20%7C%20y-%5Chat%7By%7D%20%5Cright%20%7C)
<br />
with N representing the total number of data points, y being the actual output and y hat representing the predicted output.

## Root Mean Squared Log Error
RSMLE is just the log of the RMSE
<br />
<br />
![RMSLE](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7BRMSLE%7D%20%3D%20log%28%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%20%5Csum%20%28y%20-%20%5Chat%7By%7D%29%5E%7B2%7D%7D%29)
<br />
By using the log function on the RSME, it slows down the scale of the metric.

## R Squared

## Adjusted R Squared