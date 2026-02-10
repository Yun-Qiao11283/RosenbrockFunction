import numpy as np
from Tools.RosenbrockFunction import RosenbrockFunction
from Tools.Newton import Newton
from Tools.Steepest import Steepest

x_0  = np.array([1,31,17,4,9])
print(x_0)
x, i = Newton(RosenbrockFunction, x_0)
print(f"Newton method solve:{x}")
print(f"Number of iterations: {i}")

x_1, i_1 = Steepest(RosenbrockFunction, x_0)
print(f"Steepest method solve:{x_1}")
print(f"Number of iterations: {i_1}")

