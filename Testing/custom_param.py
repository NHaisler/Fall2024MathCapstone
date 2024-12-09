import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# Define parameters for the Lorenz attractor
sigma = 10.0
beta = 8/3
rho = 28.0

# Lorenz system equations
def lorenz_system(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Generate time points
dt = 0.01
t = np.arange(0, 10, dt)

# Initial state
initial_state = [-8.0, 8.0, 27.0]

# Integrate the Lorenz equations using Euler's method
def generate_data(initial_state, t):
    state = np.array(initial_state)
    data = np.zeros((len(t), len(initial_state)))
    for i in range(len(t)):
        data[i] = state
        state += lorenz_system(t[i], state) * dt
        print(state)
    return data

# Generate the original data
X = generate_data(initial_state, t)

# Fit the model using SINDy
identity = ps.IdentityLibrary()
poly     = ps.PolynomialLibrary(degree=3)
fourier  = ps.FourierLibrary()

#mylib    = identity
#mylib    = fourier

#mylib    = identity * fourier
mylib    = identity* poly
#mylib    = identity* poly + identity * fourier
#mylib    = poly

model = ps.SINDy(feature_library=mylib, feature_names=["x", "y", "z"])
model.fit(X, t=t)

print("Functions:", model.get_feature_names())


# Predict the derivatives using the identified model
X_dot = model.predict(X)

# Simulate the predicted dynamics using the identified model
def simulate_dynamics(initial_state, t, model, dt):
    state = np.array(initial_state)
    predicted_data = np.zeros((len(t), len(initial_state)))
    for i in range(len(t)):
        predicted_data[i] = state
        state += model.predict(state.reshape(1, -1)).flatten() * dt
    return predicted_data

def runge_kutta_step(state, model, dt):
    """One step of the 4th-order Runge-Kutta method."""
    k1 = model.predict(state.reshape(1, -1)).flatten()
    k2 = model.predict((state + 0.5 * dt * k1).reshape(1, -1)).flatten()
    k3 = model.predict((state + 0.5 * dt * k2).reshape(1, -1)).flatten()
    k4 = model.predict((state + dt * k3).reshape(1, -1)).flatten()
    
    # Update state using the weighted average of the k values
    new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return new_state

def simulate_dynamics_rk(initial_state, t, model, dt): 
    state = np.array(initial_state)
    predicted_data = np.zeros((len(t), len(initial_state)))
    
    for i in range(len(t)):
        predicted_data[i] = state
        state = runge_kutta_step(state, model, dt)  # Use Runge-Kutta for updating state
    
    return predicted_data



# Generate predicted data
predicted_X = simulate_dynamics(initial_state, t, model, dt)
#predicted_X  = simulate_dynamics_rk(initial_state, t, model, dt)
print(predicted_X)

# Plot both the original and predicted results in 3D space
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Original data
ax.plot(X[:, 0], X[:, 1], X[:, 2], color='blue', label='Original Dynamics', linestyle='--')

# Predicted data
ax.plot(predicted_X[:, 0], predicted_X[:, 1], predicted_X[:, 2], color='red', label='Predicted Dynamics')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Original vs Predicted Lorenz Attractor Dynamics')
ax.legend()

# Show plot
plt.show()

model.print()