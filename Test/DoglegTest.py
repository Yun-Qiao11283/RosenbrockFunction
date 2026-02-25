import numpy as np

from Functions.RosenbrockFunction import RosenbrockFunction
from Tools.Trust_Region import TrustRegion

x_0  = np.array([1,31,17,4,9])
print(x_0)
x, i = TrustRegion.Dogleg_Method(RosenbrockFunction.f, RosenbrockFunction.gradient, RosenbrockFunction.hessian, x_0)
print(f"Dogleg method solve:{x}")
print(f"Number of iterations: {i}")