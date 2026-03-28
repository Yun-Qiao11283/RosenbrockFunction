import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from Tools.LogReg import LogisticRegression

X,y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42 )
x1 = X[:, 0]
x2 = X[:, 1]
x3 = (x1**2 + x2**2).reshape(-1,1)
X_circle = np.hstack([X, x3])
model = LogisticRegression()
model.Hessian_fit(X_circle, y)

w1, w2, w3 = model.weights[0], model.weights[1], model.weights[2]
b = model.bias
print(f"Final Weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")
print(f"Final Bias: b={b:.4f}")

plt.figure(figsize=(10, 8))

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0', alpha=0.6, edgecolors='k')

plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1', alpha=0.6, edgecolors='k')

plt.title(r"Logistic Regression Decision Boundary: $w_1 x_1 + w_2 x_2 + w_3 (x_1^2 + x_2^2) + b = 0$", fontsize=14)
plt.xlabel(r"Feature $x_1$", fontsize=12)
plt.ylabel(r"Feature $x_2$", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(0, color='gray', linewidth=0.5, ls='--')
plt.axvline(0, color='gray', linewidth=0.5, ls='--')
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = (b + w1 * xx + w2 * yy + w3 * (xx**2 + yy**2))
plt.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2)
plt.show()