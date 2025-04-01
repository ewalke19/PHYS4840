import numpy as np
import matplotlib.pyplot as plt 
import math

data = np.loadtxt('rk2_results.dat', skiprows=1)

def RungeKutta2(f, x0, t0, t_end, dt):
	t_values = np.arange(t0, t_end + dt, dt)
	x_values = np.zeros(len(t_values))
	x_values[0] = x0

	for i in range(1, len(t_values)):
		t = t_values[i - 1]
		x = x_values[i - 1]
		k1 = dt * (-x**3 + sin(t))
		k2 = dt * (-(x + 0.5*k1)**3 + sin(t +0.5*dt))
		x_values[i] = x + k2
	return t_values, x_values


def differential_eq(x, t):
	my_eqn = -x**3 + sin(t)
	return my_eqn

t_values = data[:,0]
x_values = data[:,1]
# Initial conditions
t0 = 0.0
x0 = 1.0
t_end = 10.0
dt = 0.1

# Solve using RK2 method
#t_values, x_values = RungeKutta2(my_eqn, x0, t0, t_end, dt)

# Plotting the solution
plt.figure(figsize=(8, 5))
plt.plot(t_values, x_values, label="RK2 solution", color="b")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("RK2 Solution for dx/dt = -x^3 + sin(t)")
plt.grid(True)
plt.legend()
plt.show()