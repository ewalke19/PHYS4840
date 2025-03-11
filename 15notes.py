from math import exp
from math import log
import numpy as np

x=0.5
for i in range(10):
	x=2-exp(-x)
	print(x)



for i in range(30):
	x=exp(1-x**2)
	print(x)


for i in range(100):
	x=np.sqrt(1.0-np.log(x))
	print(x)