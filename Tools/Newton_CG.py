import numpy as np

def CG_Steihaug(grad, hess, delta):
    z = np.zeros_like(grad)
    r = grad
    d = -r
    epsilon = min(0.5, np.sqrt(np.linalg.norm(r))) * np.linalg.norm(r)

    if np.linalg.norm(r) < epsilon:
        return z, "Met stopping test"

    for j in range(len(grad)):
        dBd = d.T @ hess @ d
        if dBd <= 0:
            #Find τ such that pk = zj + τdj minimizes mk(pk) in (4.5) and satisfies ||pk|| = delta_k
            #||z||^2 + 2*tau*(z^T*d) + tau^2*||d||^2 = delta^2
            a = np.dot(d,d)
            b = 2 * np.dot(z, d)
            c = np.dot(z, z) - delta**2
            tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            return z + tau * d, "Negative curvature"
        alpha = np.dot(r, r) / dBd
        z_next = z + alpha * d
        if np.linalg.norm(z_next) >= delta:
            # Find τ >= 0 such that pk = zj + τdj minimizes mk(pk) in (4.5) and satisfies ||pk|| = delta_k
            # ||z||^2 + 2*tau*(z^T*d) + tau^2*||d||^2 = delta^2
            a = np.dot(d, d)
            b = 2 * np.dot(z, d)
            c = np.dot(z, z) - delta**2
            tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            return z + tau * d, "Trust-region boundary"
        r_next = r + alpha * (hess @ d)
        if np.linalg.norm(r_next) < epsilon:
            return z_next, "Met stopping test"
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        d = -r_next + beta * d
        z = z_next
        r = r_next
    return z, "Max iterations"