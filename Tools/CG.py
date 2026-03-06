import numpy as np

def CG(A, x_0, b, t = 1e-6, max_iter = 10000):
    x = x_0.copy()
    r = A @ x_0 - b
    p = -r
    k = 0

    while(np.linalg.norm(r) >= t):
        Ap = A @ p
        r_dot_r = np.dot(r, r)
        #5.24a
        alpha = r_dot_r / np.dot(p, Ap)
        #5.24b
        x = x + alpha * p
        #5.24c
        r_new = r + alpha * Ap
        #5.24d
        beta = np.dot(r_new, r_new) / r_dot_r
        #5.24e
        p = -r_new + beta * p
        #5.24f
        r = r_new
        k += 1

        if k >= max_iter:
            print(f"Warning: {max_iter}，no converge。")
            break

    return x, k




