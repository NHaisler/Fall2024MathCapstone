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


# Generate measurement data
dt = .18

t_train = np.arange(0, 10, dt)

x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

#Instantiate and fit the SINDy model
model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()




# # Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, 10, dt)
x0_test = np.array([8, 7, 15])
t_test_span = (t_test[0], t_test[-1])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

print(t_test)


# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, t=dt))

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test)



# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
plt.show()




def get_random_num(x, y, z):
    x_large = np.float64(10**14) * np.float64(x)
    y_large = np.float64(10**14) * np.float64(y)
    z_large = np.float64(10**14) * np.float64(z)

    v_large = x_large + y_large + z_large
    K = v_large % 256

    return K


print("Lorenz Random Numbers:")
for i in range(0, 10):
    r = str(int(get_random_num(x_train[i][0], x_train[i][1], x_train[i][2]))).rjust(3, "0")
    bin_form = str(bin(int(r)))[2:].rjust(8, "0")
    print("Random Int:", r, "In Binary:", bin_form)

print("Sampled Lorenz Random:")
for i in range(0, 10):
    r = str(int(get_random_num(x_dot_test_predicted[i][0], x_dot_test_predicted[i][1], x_dot_test_predicted[i][2]))).rjust(3, "0")
    bin_form = str(bin(int(r)))[2:].rjust(8, "0")
    print("Random Int:", r, "In Binary:", bin_form)