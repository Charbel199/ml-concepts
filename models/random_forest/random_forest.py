from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
n = 100
model = RandomForestClassifier(n_estimators=n)
model.fit(digits.data[:-1], digits.target[:-1])
pred = model.predict(digits.data[-1:])

plt.matshow(digits.images[-1])
plt.show()
print(f"Predicted digit: {pred}")
