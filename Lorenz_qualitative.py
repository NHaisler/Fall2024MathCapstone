import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define Lorenz attractor equations
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

    
def analyze_quadrant_behavior_with_loops(data):
    """
    Analyzes the behavior of the system based on provided x, y, z triplets.
    
    Args:
    - data (array): A 2D numpy array of shape (n, 3), where each row is an [x, y, z] triplet.

    Returns:
    - List of behaviors as 'Left' or 'Right' depending on the x-values.
    """
    x_values = data[:, 0]  # Extract x values
    y_values = data[:, 1]  # Extract y values

    behaviors = []
    last_quadrant = None
    last_y_peak = None

    for i in range(1, len(x_values) - 1):
        # Check for local maxima or minima in y to detect loops
        if y_values[i - 1] < y_values[i] > y_values[i + 1] or y_values[i - 1] > y_values[i] < y_values[i + 1]:
            current_quadrant = "Left" if x_values[i] < 0 else "Right"

            # Detect transitions and consecutive stays
            if current_quadrant != last_quadrant or (behaviors and behaviors[-1] == current_quadrant):
                behaviors.append(current_quadrant)
                last_quadrant = current_quadrant
                last_y_peak = y_values[i]

    return behaviors

def count_consecutive_runs(behaviors):
    """
    Count how many times each behavior (Left/Right) appears consecutively.
    
    Args:
    - behaviors (list): List of behaviors ('Left' or 'Right').

    Returns:
    - List of counts for consecutive behaviors.
    """
    counts = []
    current_count = 1  # Start with 1 for the first occurrence

    for i in range(1, len(behaviors)):
        if behaviors[i] == behaviors[i - 1]:
            current_count += 1  # Continue the current run
        else:
            counts.append(current_count)  # End the current run
            current_count = 1  # Reset for the new run

    counts.append(current_count)  # Add the last run count
    return counts

# # Example usage
# initial_state = [-.14159, -.2718, .87123]
# t_span = (0, 25)
# t_eval = np.linspace(0, 25, 10000)
# sigma = 10
# beta = 8 / 3
# rho = 28

# # Solve the Lorenz system
# sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, beta, rho), t_eval=t_eval)
# data = np.vstack((sol.y[0], sol.y[1], sol.y[2])).T  # Stack x, y, z values as 2D array

# # Now analyze behavior with the provided x, y, z triplet data
# behaviors = analyze_quadrant_behavior_with_loops(data)
# print(len(behaviors))  # Print the number of quadrant changes
# print(count_consecutive_runs(behaviors))  # Print consecutive behavior counts



def behaviors_to_string(behaviors):
    temp = ""
    for item in behaviors:
        if item == "Left":
            temp += "0"
        else:
            temp += "1"
    return temp