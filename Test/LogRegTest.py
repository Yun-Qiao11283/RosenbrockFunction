import numpy as np
import matplotlib.pyplot as plt
from Tools.LogReg import LogisticRegression

np.random.seed(42)

X_class0 = np.random.randn(50,2) + np.array([-2, -2])
X_class1 = np.random.randn(50,2) + np.array([2,2])
X = np.vstack((X_class0, X_class1))
y = np.array([0]*50 + [1]*50)

model = LogisticRegression()
model.fit(X, y)

w1, w2 = model.weights[0], model.weights[1]
b = model.bias
print(f"Final Weights: w1={w1:.4f}, w2={w2:.4f}")
print(f"Final Bias: b={b:.4f}")

plt.figure(figsize=(10, 8))

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0', alpha=0.6, edgecolors='k')

plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1', alpha=0.6, edgecolors='k')

if w2 != 0:
    x1_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2_line = (-w1 * x1_line - b) / w2
    plt.plot(x1_line, x2_line, color='green', linestyle='-', linewidth=2, label='Decision Boundary')
else:
    x1_line_val = -b / w1
    plt.axvline(x=x1_line_val, color='green', linestyle='-', linewidth=2, label='Decision Boundary')



plt.title(r"Logistic Regression Decision Boundary: $w_1 x_1 + w_2 x_2 + b = 0$", fontsize=14)
plt.xlabel(r"Feature $x_1$", fontsize=12)
plt.ylabel(r"Feature $x_2$", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(0, color='gray', linewidth=0.5, ls='--')
plt.axvline(0, color='gray', linewidth=0.5, ls='--')
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

plt.show()