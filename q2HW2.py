import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-100, 100, 400)
y = x**4

fig, axs = plt.subplots(1, 3, figsize=(15, 5))


axs[0].plot(x, y)
axs[0].set_xscale('linear')
axs[0].set_yscale('linear')
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].set_title("Plot of x^4")
axs[0].grid(True)


axs[1].plot(x,y)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid(True)
axs[1].set_title("Log plot")


x2 = np.log10(x)
y2 = np.log10(y)

axs[2].plot(x2, y2)
axs[2].set_xscale('linear')
axs[2].set_yscale('linear')
axs[2].grid(True)


plt.show()