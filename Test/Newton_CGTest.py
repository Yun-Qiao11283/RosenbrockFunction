import numpy as np

from Functions.Function42 import Function42
from Tools.Trust_Region import TrustRegion

np.random.seed(42)
x_0 = np.random.randn(20)
print(x_0)
x, i = TrustRegion.Newton_CG_Method(Function42.f, Function42.gradient, Function42.hessian, x_0)
print(f"Newton method solve:{x}")
print(f"Number of iterations: {i}")