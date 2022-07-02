# Cross Entropy Loss
Cross entropy loss is used when the classification problem has more than 2 classes.

Here is the loss function equation:

![ce](../../assets/CategoricalCrossEntropy.png)

Output Layer Configuration: One node for each class using the **softmax** activation function.

Argmax could be used after the softmax activation function to determine the maximum likelihood class.