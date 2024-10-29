import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pysindy.utils import lorenz
import pysindy as ps
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


def generate_points(num_points = 10, sampled = False):
    dt = .18
    sim_time = num_points*dt
    t_train = np.arange(0, sim_time, dt)
    x0_train = [-8, 8, 27]
    t_train_span = (t_train[0], t_train[-1])
    x_train = solve_ivp(
        lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T

    if not sampled:  
        return x_train
    else:
        #Instantiate and fit the SINDy model
        model = ps.SINDy()
        model.fit(x_train, t=dt)

        #Evolve the Lorenz equations in time using a different initial condition
        t_test = np.arange(0, sim_time, dt)
        x0_test = np.array([8, 7, 15])
        t_test_span = (t_test[0], t_test[-1])
        x_test = solve_ivp(
            lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
        ).y.T

        # Predict derivatives using the learned model
        return model.predict(x_test)

def get_random_num(x, y, z, length):
    x_large = np.float64(10**14) * np.float64(x)
    y_large = np.float64(10**14) * np.float64(y)
    z_large = np.float64(10**14) * np.float64(z)

    v_large = x_large + y_large + z_large
    K = v_large % 2**length

    return K

def lorenz_random_number(n, sampled = False, length = 8):
    
    if not sampled:
        x_train = generate_points(n)
        for i in range(n):
            r = str(int(get_random_num(x_train[i][0], x_train[i][1], x_train[i][2], length))).rjust(3, "0")
            yield str(bin(int(r)))[2:].rjust(length, "0")
    else:
        x_dot_test_predicted = generate_points(n, sampled)
        for i in range(n):
            r = str(int(get_random_num(x_dot_test_predicted[i][0], x_dot_test_predicted[i][1], x_dot_test_predicted[i][2], length))).rjust(3, "0")
            yield str(bin(int(r)))[2:].rjust(length, "0")



#Might need to take derivate, since the initial conditions will move around the quandrants and the system doesn't have a clear left and rights
def orbit_sampling(num_points):
    dt = .18
    sim_time = num_points*dt

    t_train = np.arange(0, sim_time, dt)
    x0_train = [-8, 8, 27]
    t_train_span = (t_train[0], t_train[-1])
    x_train = solve_ivp(
        lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T
    print(x_train)
    model = ps.SINDy()
    model.fit(x_train, t=dt)

    #Evolve the Lorenz equations in time using a different initial condition
    t_test = np.arange(0, sim_time, dt)
    x0_test = np.array([8, 7, 15])
    t_test_span = (t_test[0], t_test[-1])
    x_test = solve_ivp(
        lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
    ).y.T

    # Predict derivatives using the learned model
    print(model.predict(x_test))
