# K-Nearest Neighbour

A KNN model calculates the distances between a point and its k nearest neighbours to determine its value / class.

During training/fit process, KNN arranges the data (sort of indexing process) in order to find the closest neighbors efficiently during the inference phase.

Some common problems with choosing a K value:
* Choosing a large K value might lead to underfitting (Model does not properly learn)
* Choosing a small K value might lead to overfitting