# Squared Hinge Loss
The squared hinge is used to draw an accurate decision boundary which penalizes large errors more significantly than smaller errors. (Usually use with SVMs)

Here is the loss function equation:

![shl](../../assets/squared_hinge_loss.png)

Output Layer Configuration: One node with a **tanh** activation unit. (Target values are in the set {-1, 1})