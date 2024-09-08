import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps

# Define the Lorenz system
def lorenz(t, state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Simulate the Lorenz system using solve_ivp
t_span = (0, 25)  # Time span for the integration
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is evaluated
initial_state = [1.0, 1.0, 1.0]  # Initial conditions

# Solve the system of ODEs
solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extract the time and solution
t = solution.t
X = solution.y.T  # Transpose the solution to get (n_samples, n_features)

# Use PySINDy to identify the governing equations
model = ps.SINDy()
model.fit(X, t=t)  # Fit the model to the data
model.print()      # Print the discovered equations

# Plot the Lorenz attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_title('Lorenz Attractor')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()




def system(t, state):
    x0, x1, x2 = state
    dx0_dt = -9.864 * x0 + 9.864 * x1
    dx1_dt = -0.213 + 26.540 * x0 - 0.712 * x1 - 0.958 * x0 * x2
    dx2_dt = 0.122 - 2.621 * x2 + 0.980 * x0 * x1
    return [dx0_dt, dx1_dt, dx2_dt]

# Set initial conditions
initial_state = [1.0, 0.0, 0.0]  # Initial values for x0, x1, x2

# Time span for the simulation
t_span = (0, 25)  # Start at t=0, end at t=25
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is evaluated

# Solve the system of ODEs
solution = solve_ivp(system, t_span, initial_state, t_eval=t_eval)

# Extract time and solutions
t = solution.t
x0, x1, x2 = solution.y

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, x0, label="x0(t)", color="blue")
plt.plot(t, x1, label="x1(t)", color="red")
plt.plot(t, x2, label="x2(t)", color="green")
plt.title("Dynamics of the System of ODEs")
plt.xlabel("Time (t)")
plt.ylabel("State Variables")
plt.legend()
plt.grid(True)
plt.show()

