from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
model = GaussianNB()
model.fit(digits.data[:-1], digits.target[:-1])
pred = model.predict(digits.data[-1:])

plt.matshow(digits.images[-1])
plt.show()
print(f"Predicted digit: {pred}")
