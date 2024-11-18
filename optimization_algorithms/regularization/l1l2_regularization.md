# L1  regularization

L1 regularization, also known as Lasso regularization, is a technique used in machine learning to prevent overfitting.
It adds a term to the loss function that penalizes the absolute value of the magnitude of the model's weights.
The regularization term is typically a multiple of the L1-norm of the weights vector, often denoted by lambda.

The effect of L1 regularization is to push the model's weights towards zero, which has the effect of shrinking the less important feature's coefficient to zero.

This can be useful for feature selection, as it tends to give sparse solutions, with many weights equal to zero. This can be contrasted with L2 regularization(Ridge regularization),
which adds a term to the loss function that penalizes the square of the magnitude of the model's weights, and encourages small weights, but not zero weights.

![l1](docs/L1Cost.png)


## Reference(s)
[L1 Regularization](https://theaisummer.com/regularization/)


# L2 regularization

L2 regularization simply consists of adding a regularization value to our cost function.

![l2](docs/L2Cost.png)

As we can see in the L2 regularization cost, it penalizes high values in the model's weights making it less
prone to overfitting.

## Reference(s)
[L2 Regularization for Learning Kernels](https://arxiv.org/abs/1205.2653)
[L1 regularization.md](L1%20Regularization.md)


# Elastic Net regularization

Elastic Net regularization is a combination of both L1 and L2 regularization techniques.
It is a linear regression model with both L1 and L2 regularization terms added to the loss function.
The regularization term is typically a combination of the L1-norm and the L2-norm of the weights vector, often denoted by lambda and alpha respectively.

The Elastic Net regularization method seeks to balance the strengths of L1 and L2
regularization by adding both terms to the loss function. The L1 term results in feature selection and the L2
term will shrink the less important feature's coefficient. L1 regularization will give sparse solutions,
with many weights equal to zero and L2 regularization will encourage small weights, but not zero weights.

By combining these two penalties, Elastic Net regularization can often give better results than either L1 or L2 regularization alone.

More specifically the penalty term is as follows:

![elastic net](docs/ElasticNetCost.png)



Elastic Net is parameterized by two hyperparameters:

- alpha, which controls the L1 regularization term and
- l1_ratio, which controls the balance between L1 and L2 regularization.

Typically, when l1_ratio=1, it is L1 regularization, and when l1_ratio=0, it is L2 regularization, and when 0 < l1_ratio < 1, it is Elastic Net regularization.

In summary, Elastic Net regularization is a technique that combines L1 and L2 regularization techniques in order to balance their strengths and to overcome the limitations of L1 and L2 regularization methods.



## Reference(s)
[Elastic Net Regularization](https://theaisummer.com/regularization/)