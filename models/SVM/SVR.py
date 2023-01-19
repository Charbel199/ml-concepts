from sklearn.svm import SVR
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(digits.data[:-1], digits.target[:-1])
pred = model.predict(digits.data[-1:])

plt.matshow(digits.images[-1])
plt.show()
print(f"Predicted digit: {pred}")
