import numpy as np

from Tools.Dogleg import Dogleg


class TrustRegion:
    @staticmethod
    def Dogleg_Method(f, grad, hess, x0, delta_hat=1.0, delta_0=0.2, eta=0.15):
        x = x0
        delta = delta_0

        for k in range(100):
            g = grad(x)
            B = hess(x)

            if np.linalg.norm(g) < 1e-6:  # Convergence
                break

            # 1. Find pk
            p = Dogleg(g, B, delta)

            # 2. Calculate rho
            actual_red = f(x) - f(x + p)
            predicted_red = -(np.dot(g, p) + 0.5 * np.dot(p, np.dot(B, p)))

            rho = actual_red / predicted_red

            # 3. Update radius delta
            if rho < 0.25:
                delta = 0.25 * delta
            elif rho > 0.75 and np.isclose(np.linalg.norm(p), delta):
                delta = min(2 * delta, delta_hat)

            # 4. Update x
            if rho > eta:
                x = x + p

        return x, k