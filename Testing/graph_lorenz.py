import numpy as np
import matplotlib.pyplot as plt
import pysindy as SINDy

# Define the Lorenz system
def lorenz_system(t, state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Simulate the Lorenz system
def simulate_lorenz(dt=0.01, t_max=50):
    t = np.arange(0, t_max, dt)
    state = np.zeros((len(t), 3))
    state[0] = [1.0, 1.0, 1.0]  # Initial condition

    for i in range(1, len(t)):
        state[i] = state[i-1] + lorenz_system(t[i-1], state[i-1]) * dt
    
    return t, state

# Run the simulation
dt = 0.01
t_max = 50
t, state = simulate_lorenz(dt, t_max)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state[:, 0], state[:, 1], state[:, 2])
ax.set_title('Lorenz System')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Apply PySINDy to identify the system dynamics
# Reshape the state for SINDy
X = state[:-1]  # Use all but the last state
dX = np.diff(state, axis=0) / dt  # Compute derivatives

# Fit the SINDy model
model = SINDy.SINDy()
model.fit(X, dX)
model.print()

# Predict using the identified model
X_pred = model.predict(X)

# Plot the original vs predicted
plt.figure()
plt.plot(X[:, 0], label='Original X')
plt.plot(X_pred[:, 0], label='Predicted X', linestyle='dashed')
plt.legend()
plt.title('Original vs Predicted X using PySINDy')
plt.xlabel('Time Step')
plt.ylabel('Value of X')
plt.show()
