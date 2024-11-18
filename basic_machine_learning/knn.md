# K-Nearest Neighbour

A KNN model calculates the distances between a point and its 'K' Nearest Neighbours to determine its value / class.

During training/fit process, KNN arranges the data (in a sort of indexing process) in order to find the closest neighbors efficiently during the inference phase.

Some common problems with choosing a K value:
* Choosing a large K value might lead to underfitting (Model does not properly learn)
* Choosing a small K value might lead to overfitting


Here is a simple KNN [code sample](../basic_machine_learning/knn.py)

## Reference(s)
[KNN Model-Based Approach in Classification](https://www.researchgate.net/publication/2948052_KNN_Model-Based_Approach_in_Classification)