import numpy as np
import matplotlib.pyplot as plt

# ================= C2 =================
N_values = [8, 16, 32, 64, 128, 256]
solutions = {}
h_values = []
L_inf_errors = []
L2_errors = []

for N in  N_values:
    h = 1.0 / N
    h_values.append(h)

    x = np.linspace(h, 1 - h, N - 1)
    main_diag = (2.0 / h ** 2) * np.ones(N - 1)
    off_diag = (-1.0 / h ** 2) * np.ones(N - 2)
    A_h = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    b = (np.pi**2) * np.sin(np.pi * x)
    u_h = np.linalg.solve(A_h, b)

    solutions[N] = {'x': x, 'u_h': u_h}

    # ================= C3 =================
    u_exact = np.sin(np.pi * x)
    e_h = u_exact - u_h
    L_inf = np.max(np.abs(e_h))
    L2 = np.sqrt(h * np.sum(e_h ** 2))

    L_inf_errors.append(L_inf)
    L2_errors.append(L2)

    print(f"N = {N}, u = {u_h}, L_inf = {L_inf}, L2 = {L2}")

# ================= C5 =================
for i in range(len(N_values) - 1):
    h1, h2 = h_values[i], h_values[i + 1]

    rate_inf = np.log(L_inf_errors[i] / L_inf_errors[i + 1]) / np.log(h1 / h2)
    rate_L2 = np.log(L2_errors[i] / L2_errors[i + 1]) / np.log(h1 / h2)

    interval_str = f"{N_values[i]} -> {N_values[i + 1]}"
    print(f"{interval_str:<15} | {rate_inf:<15.4f} | {rate_L2:<15.4f}")

# ================= C4 =================
plt.figure(figsize=(8, 6))

plt.loglog(h_values, L_inf_errors, 'o-', linewidth=2, label='$L_\infty$ Error')
plt.loglog(h_values, L2_errors, 's-', linewidth=2, label='$L_2$ Error')

C_ref = L_inf_errors[0] / (h_values[0]**2)
ref_line = [C_ref * (h**2) for h in h_values]
plt.loglog(h_values, ref_line, 'k--', linewidth=1.5, label='$\mathcal{O}(h^2)$ Reference')

plt.xlabel('Grid spacing $h$', fontsize=12)
plt.ylabel('Error Norm', fontsize=12)
plt.title('Convergence Study of Finite Difference Method', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12)
plt.gca().invert_xaxis() # 通常习惯将 h 从大到小排列 (从左往右网格越来越密)

plt.tight_layout()
plt.show()
