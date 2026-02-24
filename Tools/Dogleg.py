import numpy as np

def Dogleg(g, B, delta):
    try:
        pB = -np.linalg.solve(B, g)
    except np.linalg.LinAlgError:
        n = len(g)
        eigvals = np.linalg.eigvalsh(B)
        lambda_min = eigvals[0]
        curr_lambda = max(0, -lambda_min + 1e-3)
        for _ in range(10):
            # 1. Cholesky: B + lambda*I = R^T R
            try:
                R = np.linalg.cholesky(B + curr_lambda * np.eye(n)).T
                # 2. Solve R^T R p = -g 和 R^T q = p
                p = np.linalg.solve(R.T @ R, -g)
                norm_p = np.linalg.norm(p)

                # ||p|| close to delta
                if abs(norm_p - delta) < 1e-4 * delta:
                    break

                # 3. find q
                q = np.linalg.solve(R.T, p)
                norm_q = np.linalg.norm(q)

                # 4. Update lambda
                curr_lambda = curr_lambda + (norm_p / norm_q) ** 2 * ((norm_p - delta) / delta)

            except np.linalg.LinAlgError:
                # lambda too small
                curr_lambda = max(1.1 * curr_lambda, 1e-3)

        pB = p

    norm_pB = np.linalg.norm(pB)

    #Case 1: ||pB|| <= delta
    if norm_pB <= delta:
        return pB

    #Calculate pU
    denom = np.dot(g, np.dot(B, g))
    #Negative curvation, return Cauchy point on boundary
    if denom <= 0:
        pU = -(delta / np.linalg.norm(g)) * g
        return pU

    pU = -(np.dot(g, g) / denom) * g
    norm_pU = np.linalg.norm(pU)
    # Case 2: ||pU|| > delta, return cauchy point
    if norm_pU >= delta:
        return (delta / norm_pU) * pU

    # Case 3：Solve the quadratic equation for the slope to find the intersection points of the broken line.
    # Equation：||pU + (tau - 1)(pB - pU)||^2 = delta^2
    v = pB - pU
    a = np.dot(v, v)
    b = 2 * np.dot(pU, v)
    c = np.dot(pU, pU) - delta ** 2

    # solve t = tau - 1
    t = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return pU + t * v

