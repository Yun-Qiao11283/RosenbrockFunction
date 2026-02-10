import numpy as np

def Newton(f,x_0, tol=1e-5, max_iter=100):
    x = np.array(x_0, dtype=float)
    for i in range(max_iter):
        grad = f.gradient(x)
        hess = f.hessian(x)
        if np.linalg.norm(grad) < tol:
            return x, i
        try:
            p = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            p = -grad

        x += p

        if np.linalg.norm(p) < tol:
            return x, i
    return x, max_iter