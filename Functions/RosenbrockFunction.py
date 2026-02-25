import numpy as np

class RosenbrockFunction:
    #F(x) = sum_1^{n-1}(100*(x_i^2-x_{i+1})^2+(x_i-1)^2)
    @staticmethod
    def f(x):
        x = np.array(x)
        return sum(100 * (x[:-1]**2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    #F_1 = 400*(x_1^2-x_2)*x_1+2(x_1-1)
    #F_i = 400*(x_i^2-x_{i+1})*x_1+2(x_i-1)-200*(x_{i-1}^2-x_i)
    #F_n = -200(x_{n-1}^2-x_n)
    @staticmethod
    def gradient(x):
        x = np.array(x)
        g = np.zeros_like(x)


        xi = x[:-1]
        xi_next = x[1:]

        term1 = 400 * xi * (xi**2 - xi_next) + 2 * (xi - 1)
        term2 = -200 * (xi**2 - xi_next)

        g[:-1] += term1
        g[1:] += term2
        return g

    #F_{1,1} = 400(3x_1^2 - x_2) + 2
    #F_{i,i} = 400(3x_i^2 - x_{i+1}) + 2 + 200
    #F_{n,n} = 200
    #F_{i, i+1} = F_{i+1, i} = -400x_i
    @staticmethod
    def hessian(x):
        x = np.array(x)
        n = len(x)
        H = np.zeros((n, n))


        H[0,0] = 400 * (3 * x[0]**2 - x[1]) + 2
        for i in range(1,n - 1):
            H[i,i] = 400 * (3 * x[i]**2 - x[i + 1]) + 202
        H[n - 1, n - 1] = 200

        for i in range(n - 1):
            val = -400 * x[i]
            H[i, i+1] = val
            H[i+1, i] = val

        return H


