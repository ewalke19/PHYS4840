import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# some data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 2, 1, 3, 7, 8, 7, 9, 10])  

# Define fine-grained x-values for interpolation
x_domain = np.linspace(min(x), max(x), 200)
print(x_domain)

# Linear Interpolation
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_domain)

# Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)
y_cubic = cubic_spline(x_domain)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points', zorder=3)
plt.plot(x_domain, y_linear, '--', label='Linear Interpolation', linewidth=2)
plt.plot(x_domain, y_cubic, label='Cubic Spline Interpolation', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs. Cubic Spline Interpolation')
plt.grid(True)
plt.show()


# adding more data points, the two lines look more like each other. 




import numpy as np
import matplotlib.pyplot as plt
from math  import tanh, cosh

import sys
sys.path.append('../')
import my_functions_lib as mfl

## compute the instantaneous derivatives
## using the central difference approximation
## over the interval -2 to 2

x_lower_bound = -2.0
x_upper_bound = 2.0

N_samples = 100

#####################
#
# Try different values of h
# What did we "prove" h should be
# for C = 10^(-16) in Python?
#
#######################
h = 10**-10 ## what goes here?
h2 = 2
h3 = 1
h4 = 1**-16
h5 = 10**-3

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values = []
for x in xdata:
	central_difference = ( mfl.f(x + 0.5*h) - mfl.f(x - 0.5*h) ) / h
	central_diff_values.append(central_difference)
central_diff_values2 = []
for x in xdata:
	central_difference2 = ( mfl.f(x + 0.5*h2) - mfl.f(x - 0.5*h2) ) / h2
	central_diff_values2.append(central_difference2)
central_diff_values3 = []
for x in xdata:
	central_difference3 = ( mfl.f(x + 0.5*h3) - mfl.f(x - 0.5*h3) ) / h3
	central_diff_values3.append(central_difference3)
central_diff_values4 = []
for x in xdata:
	central_difference4 = ( mfl.f(x + 0.5*h4) - mfl.f(x - 0.5*h4) ) / h4
	central_diff_values4.append(central_difference4)
central_diff_values5 = []
for x in xdata:
	central_difference5 = ( mfl.f(x + 0.5*h5) - mfl.f(x - 0.5*h5) ) / h5
	central_diff_values5.append(central_difference5)

## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values = []
for x in xdata:
	dfdx = mfl.df_dx_analytical(x)
	analytical_values.append(dfdx)


plt.plot(xdata, analytical_values, linestyle='-', color='black', label = 'h = 10**-10')
plt.plot(xdata, central_diff_values, "*", color="green", markersize=8, alpha=0.5)
plt.plot(xdata, central_diff_values2, "-", color="red", markersize=8, alpha=0.5,  label = 'h = 2')
plt.plot(xdata, central_diff_values3, "-", color="blue", markersize=8, alpha=0.5, label = 'h = 1')
plt.plot(xdata, central_diff_values4, "-", color="orange", markersize=8, alpha=0.5, label = 'h = 1**-16')
plt.plot(xdata, central_diff_values5, "-", color="yellow", markersize=8, alpha=0.5, label = 'h = 10**-3')
#plt.savefig('numerical_vs_analytic_derivatives.png')
#plt.close()
plt.legend()
plt.show()