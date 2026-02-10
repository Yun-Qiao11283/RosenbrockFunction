import numpy as np
from scipy.optimize import minimize

from Tools.RosenbrockFunction import RosenbrockFunction

n = 5
x_0 = np.array([-1.2] * n)

#Quai-Newton(BFGS)
res_bfgs = minimize(RosenbrockFunction.f, x_0, jac=RosenbrockFunction.gradient, method='BFGS')

#Newton
res_nt = minimize(RosenbrockFunction.f, x_0, jac=RosenbrockFunction.gradient, hess=RosenbrockFunction.hessian, method='Newton-CG')

#the steepest
res_steep = minimize(RosenbrockFunction.f, x_0, jac=RosenbrockFunction.gradient, method='CG')

print(res_bfgs.x)
print(f"Number of iterations: {res_bfgs.nit}")
print(res_nt.x)
print(f"Number of iterations: {res_nt.nit}")
print(res_steep.x)
print(f"Number of iterations: {res_steep.nit}")

