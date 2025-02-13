#question one

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

data = np.loadtxt('NGC6341.dat', usecols=(8, 14, 26)) 
data2 = np.loadtxt('MIST_v1.2_feh_m1.75_afe_p0.0_vvcrit0.4_HST_WFPC2.iso.cmd', usecols=(4))

temp =data2[0]
blue = data[:, 0] 
green = data[:, 1] 
red = data[:, 2]

color = blue - red
magnitude = blue

#fig, axs = plt.subplots()
fig, axs1 = plt.subplots()

plt.figure(figsize=(8, 6)) 
axs1.scatter(color, magnitude, s=10, alpha=0.5)
axs1.set_xscale('log')
axs1.set_yscale('log')
axs1.set_xlim(0, 7)
axs1.set_ylim(14, 25)
axs1.set_xlabel('Color b-r')
axs1.set_ylabel('Magnitude b')
axs1.set_title('Color-Magnitude Diagram')
axs1.invert_yaxis() 

axs2 = axs1.twinx()
axs2.scatter(color, temp, color=red)
axs2.set_xscale('log')
axs2.set_yscale('linear')
axs2.set_ylabel('Temp')

plt.show()

#works with just axs1, when adding in axs2 a value error pops up saying x and y must be the same size





# question two
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






#question three
import matplotlib.pyplot as plt
import numpy as np
import math

data = np.loadtxt('sunspots.txt', usecols=(0,1))

x = data[:, 0] 
y = data[:, 1]

#plt.plot(x,y)
#plt.show()

yk = np.random.normal(10, 5, size=1000)
r = 5
m = -5
def average(y, r, m):
	return ((1/(2*r+1))*(yk+m))

fig, axs = plt.subplots()

x_data_subset = x[:1000]
y_data_subset = y[:1000]
average_data_subset = yk[:1000]

axs.plot(x_data_subset, average_data_subset, color = 'r')
axs.set_ylim(0,250)

#print(average)
#print(y)

axs.plot(x_data_subset, y_data_subset)
plt.show()





#question four 
git fetch origin
git checkout main
git rebase origin/main
git push origin main 

git status
git pull origin
git add "q1HW2.py"
git commit -m "Homework 2"
git push origin

rm -rf .git
rm -rf .gitignore



#question five 

