import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.betas = None

    def Hessian(self, X, y_pred):
        m = X.shape[0]
        s = (y_pred * (1 - y_pred)).reshape(-1, 1)

        #1 by 1
        d2J_db2 = np.sum(s) / m
        #m by m
        d2J_dw2 = (X.T @ (s * X)) / m
        #1 by m
        d2J_dwdb = (np.sum(s * X, axis=0)) / m

        #b, w1, w2
        row1 = np.hstack([d2J_db2, d2J_dwdb])
        row23 = np.hstack([d2J_dwdb.reshape(-1, 1), d2J_dw2])

        hessian = np.vstack([row1, row23])

        return hessian


    def Hessian_fit(self, X, y):
        y = y.reshape(-1, 1)
        num_samples, num_features = X.shape
        ones = np.ones((num_samples,1))
        X_hat = np.hstack([ones, X])
        self.betas = np.zeros(num_features + 1)
        for _ in range(self.num_iterations):
            linear_model = np.dot(X_hat, self.betas.reshape(-1,1))
            y_pred = sigmoid(linear_model).reshape(-1, 1)
            gradient = (1 / num_samples) * np.dot(X_hat.T, (y_pred - y))
            H = self.Hessian(X, y_pred)
            epsilon = 1e-4
            H_reg = H + epsilon * np.eye(H.shape[0])
            p = np.linalg.solve(H_reg, -gradient).flatten()
            self.betas += p
            if np.linalg.norm(p) < 1e-6:
                break
        self.weights = self.betas[1:]
        self.bias = self.betas[0]

    def circle_fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if np.linalg.norm(dw) < 1e-6 and abs(db) < 1e-6:
                break

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if np.linalg.norm(dw) < 1e-6 and abs(db) < 1e-6:
                break

    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X):
        y_predicted_probability = self.predict_probability(X)
        y_predicted = np.where(y_predicted_probability > 0.5, 1, 0)
        return y_predicted

