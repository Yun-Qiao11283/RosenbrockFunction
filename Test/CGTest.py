import numpy as np

from Tools.CG import CG


def generate_hilbert_matrix(n):
    """Hilbert matrix"""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1.0 / (i + j + 1)
    return A

dim = [5, 8, 12, 20]
tol = 1e-6
print(f"Tolerance: {tol}\n")
print("-" * 30)

for n in dim:
    A = generate_hilbert_matrix(n)
    b = np.ones(n)
    x0 = np.zeros(n)

    x_sol, iters = CG(A, b, x0, tol)
    print(f"Dimension n = {n:<2} | Iteration k = {iters}")
    print(f"Solution x = {x_sol}")
print("-" * 30)