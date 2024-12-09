import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import lorenz
import pysindy as ps

def original(num_points=10, iv = [-8, 8, 27]):
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = 1e-12

    sim_time = 10
    dt = 10/num_points
    t_train = np.arange(0, sim_time, dt)
    x0_train = iv
    t_train_span = (t_train[0], t_train[-1])
    x_train = solve_ivp(
        lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T

    return x_train

import numpy as np
from scipy.integrate import solve_ivp

def rossler(t, x, a=0.2, b=0.2, c=5.7):
    """Rossler attractor system."""
    dxdt = -x[1] - x[2]
    dydt = x[0] + a * x[1]
    dzdt = b + x[2] * (x[0] - c)
    return [dxdt, dydt, dzdt]

def original_rossler(num_points=10, iv=[-8, 8, 27]):
    """Solve the Rossler system."""
    integrator_keywords = {
        "rtol": 1e-12,
        "method": "LSODA",
        "atol": 1e-12
    }

    sim_time = 10
    dt = 10 / num_points
    t_train = np.arange(0, sim_time, dt)
    x0_train = iv
    t_train_span = (t_train[0], t_train[-1])

    # Solve the system using solve_ivp
    x_train = solve_ivp(
        rossler, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T

    return x_train


def henon(t, x, a=1.4, b=0.3):
    """Hénon system approximated as continuous-time ODEs."""
    dxdt = 1 - a * x[0]**2 + x[1]
    dydt = b * x[0]
    return [dxdt, dydt]

def original_henon(num_points=10, iv=[0.1, 0.1]):
    """Solve the Hénon system."""
    integrator_keywords = {
        "rtol": 1e-12,
        "method": "LSODA",
        "atol": 1e-12
    }

    sim_time = 10
    dt = 10 / num_points
    t_train = np.arange(0, sim_time, dt)
    x0_train = iv
    t_train_span = (t_train[0], t_train[-1])

    # Solve the system using solve_ivp
    x_train = solve_ivp(
        henon, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T

    return x_train

def simulate_dynamics(initial_state, num_points, model, dt, max_value=1e10):
    # Ensure the state is of type float64 to avoid casting issues
    state = np.array(initial_state, dtype=np.float64)
    predicted_data = np.zeros((num_points, len(initial_state)))  # Size based on num_points
    
    for i in range(num_points):  # Iterate based on num_points
        predicted_data[i] = state
        
        # Update state using model's prediction
        state += model.predict(state.reshape(1, -1)).flatten() * dt
        
        # Check if any state variable exceeds max_value and set it to NaN (or some other action)
        state[np.abs(state) > max_value] = np.nan
        
        # Check for NaN values and handle them (e.g., reset to zero, break, etc.)
        if np.isnan(state).any():
            print(f"Warning: State contains NaN at step {i}.")
            state = np.zeros_like(state)  # Reset state to zero, or handle appropriately
            
    return predicted_data



def sampling_points(points, ratio=0.5):
    # Initialize new_points as an empty 2D array with 0 rows and the same number of columns as points
    new_points = np.empty((0, points.shape[1]))  # (0, 3) for a 3D point array, for example
    
    i = 0
    while (len(new_points) + 1) / len(points) <= ratio:
        new_points = np.vstack([new_points, points[i]])  # Stack the new point as a new row
        i += 1
    
    return new_points
