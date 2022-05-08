from sklearn.neighbors import   KNeighborsClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(digits.data[:-1], digits.target[:-1])
pred = model.predict(digits.data[-1:])

plt.matshow(digits.images[-1])
plt.show()
print(f"Predicted digit: {pred}")
