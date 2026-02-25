import numpy as np

class Function42:
    @staticmethod
    def f(x):
        n = len(x)
        f = 0
        for i in range(n // 2):
            x1, x2 = x[2 * i], x[2 * i + 1]
            f += (1 - x1) ** 2 + 10 * (x2 - x1 ** 2) ** 2
        return f

    @staticmethod
    def gradient(x):
        n = len(x)
        grad = np.zeros(n)
        for i in range(n // 2):
            x1, x2 = x[2 * i], x[2 * i + 1]
            grad[2 * i] = -2 * (1 - x1) - 40 * x1 * (x2 - x1 ** 2)
            grad[2 * i + 1] = 20 * (x2 - x1 ** 2)

        return grad

    @staticmethod
    def hessian(x):
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n // 2):
            x1, x2 = x[2 * i], x[2 * i + 1]
            hess[2 * i, 2 * i] = 2 - 40 * (x2 - 3 * x1 ** 2)
            hess[2 * i, 2 * i + 1] = -40 * x1
            hess[2 * i + 1, 2 * i] = -40 * x1
            hess[2 * i + 1, 2 * i + 1] = 20
        return hess

