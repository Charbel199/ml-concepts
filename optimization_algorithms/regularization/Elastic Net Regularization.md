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

![elastic net](../docs/ElasticNetCost.png)



Elastic Net is parameterized by two hyperparameters:

- alpha, which controls the L1 regularization term and
- l1_ratio, which controls the balance between L1 and L2 regularization.

Typically, when l1_ratio=1, it is L1 regularization, and when l1_ratio=0, it is L2 regularization, and when 0 < l1_ratio < 1, it is Elastic Net regularization.

In summary, Elastic Net regularization is a technique that combines L1 and L2 regularization techniques in order to balance their strengths and to overcome the limitations of L1 and L2 regularization methods.



## Reference(s)
[Elastic Net Regularization](https://theaisummer.com/regularization/)