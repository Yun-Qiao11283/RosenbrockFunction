import numpy as np

def Steepest(f, x_0, tol=1e-5, max_iter=100000):
    x = np.array(x_0, dtype=float)
    f_prev = f.f(x)
    f_curr = f.f(x)
    for i in range(max_iter):
        grad = f.gradient(x)
        if np.linalg.norm(grad) < tol:
            return x, i
        p = -grad
        phi_p0 = np.dot(grad, p)
        if i == 0:
            alpha_init = 1.0 / np.linalg.norm(grad)
        else:
            alpha_init = max(2.0 * (f_curr - f_prev) / phi_p0, 0.1)
        f_prev = f_curr
        alpha = line_search_wolfe(f, f.gradient, x, p, alpha_init=alpha_init)
        if alpha < 1e-12:
            break
        x += alpha * p
        f_curr = f.f(x)
    return x, max_iter


def line_search_wolfe(f, grad, x, p, c1=1e-4, c2=0.9,alpha_init=0.1, a_max=50, max_iter=20):
    alpha_0 = 0
    alpha_prev = alpha_0
    alpha_curr = alpha_init

    phi = lambda a: f.f(x + a * p)

    phi_prime = lambda a: np.dot(grad(x + a * p), p)

    phi_0 = phi(0)
    phi_p0 = phi_prime(0)

    for i in range(1, max_iter):
        phi_curr_val = phi(alpha_curr)

        if (phi_curr_val > phi_0 + c1 * alpha_curr * phi_p0) or \
                (i > 1 and phi_curr_val >= phi(alpha_prev)):
            return zoom(alpha_prev, alpha_curr, phi, phi_prime, phi_0, phi_p0, c1, c2)

        phi_p_curr = phi_prime(alpha_curr)

        if abs(phi_p_curr) <= -c2 * phi_p0:
            return alpha_curr

        if phi_p_curr >= 0:
            return zoom(alpha_curr, alpha_prev, phi, phi_prime, phi_0, phi_p0, c1, c2)

        alpha_prev = alpha_curr
        alpha_curr = min(alpha_curr * 2, a_max)

    return alpha_curr


def zoom(a_lo, a_hi, phi, phi_prime, phi_0, phi_p0, c1, c2):
    for _ in range(30):
        if abs(a_lo - a_hi) < 1e-15:
            break
        alpha_j = 0.5 * (a_lo + a_hi)
        phi_j = phi(alpha_j)

        if (phi_j > phi_0 + c1 * alpha_j * phi_p0) or (phi_j >= phi(a_lo)):
            a_hi = alpha_j
        else:
            phi_p_j = phi_prime(alpha_j)
            if abs(phi_p_j) <= -c2 * phi_p0:
                return alpha_j
            if phi_p_j * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = alpha_j

    return a_lo