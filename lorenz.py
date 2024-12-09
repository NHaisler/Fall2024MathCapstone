import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lorenz system
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions
initial_conditions = [-8, 8, 27]

# Time span for the integration
t_span = (0, 20)  # From t=0 to t=50
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time steps to evaluate the solution

# Solve the system of differential equations
solution = solve_ivp(lorenz, t_span, initial_conditions, args=(sigma, rho, beta), t_eval=t_eval, rtol=1e-8)

# Plotting the solution in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.5, color='b')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.title('Lorenz Attractor')
plt.show()
