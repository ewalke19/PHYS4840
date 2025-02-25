# in my functions library
# import math

# def f(x):
# 	return 1 + 1/2(tanh(2*x))


import my_functions_lib as mfl

x = range(-2, 2)
h = 10**-10
df_dx = (mfl.f(x + h/2)-mfl.f(x - h/2))/h

print(df_dx)