# Naive Bayes
This algorithm makes an assumption that all the variables in the dataset are “Naive” (Uncorrelated).
The parameters that are learned in Naive Bayes are the prior probabilities of different classes, as well as the likelihood of different features for each class.

The Naive Bayes Formula:

$$P(C_k | \mathbf{X}) = \frac{P(\mathbf{X} | C_k) \cdot P(C_k)}{P(\mathbf{X})}$$

Where:
-  $P(C_k | \mathbf{X})$: Posterior probability of class $C_k$ given the data $mathbf{X}$.
- $P(C_k)$: Prior probability of class $C_k$.
- $P(\mathbf{X} | C_k)$: Likelihood of observing the data $\mathbf{X}$ given class $C_k$.
- $P(\mathbf{X})$: Evidence, or the overall probability of the data $\mathbf{X}$.
\end{itemize}

#### Assumptions:
1. Features in $\mathbf{X}$ are conditionally independent given the class $( C_k )$.
2. The likelihood $P(\mathbf{X} | C_k)$ is calculated as:

   $$P(\mathbf{X} | C_k) = \prod_{i=1}^{n} P(X_i | C_k)$$

   where $(X_i)$ represents individual features.

Why use Naive Bayes?
* Very simple to implement
* Doesn't require a lot of data
* Not sensitive to irrelevant features
* Highly scalable and fast

But we should keep in mind that in real life, it is almost impossible that we get a set of predictors which are completely independent.
Another thing to note is that if a given class and feature value never occur together in the training data, then the frequency-based probability estimate will be zero.

Here is a simple Naive Bayes implementation [code sample](../basic_machine_learning/naive_bayes.py)

## Reference(s)
[Naive Bayes](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_576)