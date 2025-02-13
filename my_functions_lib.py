import numpy as np

def dist_mod(d):
	mu = 5*np.log10((d*1000)/10)
	return mu