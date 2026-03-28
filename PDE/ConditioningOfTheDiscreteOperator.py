import numpy as np
import matplotlib.pyplot as plt

#============K1============
N_vals = np.array([8, 16, 32, 64, 128, 256])
h_vals = 1.0 / N_vals
kappa_vals = np.zeros(len(N_vals))

for i, N in enumerate(N_vals):
    h = h_vals[i]
    k_array = np.arange(1, N)
    lambdas = (4 / h**2) * np.sin(k_array * np.pi / (2 * N))**2
    kappa_vals[i] = np.max(lambdas) / np.min(lambdas)
    print(f"N = {N}, K2 = {kappa_vals[i]}")

#============K2============
plt.figure()
plt.loglog(h_vals, kappa_vals, 'o-', label='$\kappa_2(A_h)$')
plt.xlabel('h')
plt.ylabel('Condition Number')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()