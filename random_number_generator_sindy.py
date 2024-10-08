import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

data = (Path() / "../data").resolve()


@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


if __name__ == "testing":
    import sys
    import os

    sys.stdout = open(os.devnull, "w")


# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


# Generate measurement data
dt = 0.002

t_train = np.arange(0, t_end_train, dt)

x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T


def get_random_num(x, y, z):
    x_large = np.float64(10**14) * np.float64(x)
    y_large = np.float64(10**14) * np.float64(y)
    z_large = np.float64(10**14) * np.float64(z)

    v_large = x_large + y_large + z_large
    K = v_large % 256

    return K


for i in range(0, 10):
    r = str(int(get_random_num(x_train[i][0], x_train[i][1], x_train[i][2]))).rjust(3, "0")
    bin_form = str(bin(int(r)))[2:].rjust(8, "0")
    print("Random Int:", r, "In Binary:", bin_form)