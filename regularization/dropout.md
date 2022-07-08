# Dropout

Dropout regularization is the concept of setting a probability for each layer in a neural network for eliminating the effect of a node.

This is very useful in cases where the model relies on specific features for the task, dropout would make it rely on other less prominent features
and thus making it more general.


![dropout_concept](../assets/DropoutConcept.png)


There are several dropout functions but the most well-known one would be: "Inverted Dropout"

* Based on the keep-probability, set a weight of 0 for neurons that should be de-activated
* Divide the remaining weights by the keep-probability, this re-scales the remaining active units


## Reference(s)
[Dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
