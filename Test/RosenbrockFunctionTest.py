import numpy as np

from Tools.RosenbrockFunction import RosenbrockFunction

x_0 = np.array([1,2,3,4,9])
print(f"Function at {x_0}：")
print(f"Value: {RosenbrockFunction.f(x_0):.4f}")
print(f"Gradient: {RosenbrockFunction.gradient(x_0)}")
print(f"Hessian:\n{RosenbrockFunction.hessian(x_0)}")