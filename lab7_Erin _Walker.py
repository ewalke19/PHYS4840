#print(0.1+0.2)
#print(repr(0.1+0.2))

#from math import sqrt 

#x = 1.0 
#y = 1.0 +(1e-14)*sqrt(2)

#answer_1 = 1e14*(y-x)
#answer_2 = sqrt(2)

#print('answer1:', answer_1)
#print('answer2:', answer_2)
#print('percentage difference:', (1-(answer_2/answer_1))*100)

import time
import numpy as np
import sys
import pandas as pd

filename = 'NGC6341.dat'

start_numpy = time.perf_counter()
blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)
print("len(green): ", len(green))
end_numpy  = time.perf_counter()

start_numpy = time.perf_counter()
blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)
print("len(green): ", len(green))
end_numpy  = time.perf_counter()

print('Time to run loadtxt version: ', end_numpy-start_numpy, ' seconds')

start_parser = time.perf_counter()
blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)
print("len(green): ", len(green))
end_parser  = time.perf_counter()

blue, green, red = [], [], []
start_parser = time.perf_counter()
filename = 'NGC6341.dat'
# Open the file and read line by line
with open(filename, 'r') as file:
    for line in file:
        # Skip lines that start with '#'
        if line.startswith('#'):
            continue
        
        # Split the line into columns based on spaces
        columns = line.split()
        
        blue.append(float(columns[8]))   # Column 9 
        green.append(float(columns[14])) # Column 15 
        red.append(float(columns[26]))   # Column 27 

blue = np.array(blue)
green = np.array(green)
red = np.array(red)

print("len(green): ", len(green))
end_parser  = time.perf_counter()

print('Time to run custom parser version: ', end_parser-start_parser, ' seconds')

start_pandas = time.perf_counter()
df = pd.read_csv(filename, delim_whitespace=True, comment='#', header=None, skiprows=54)
blue = df.iloc[:, 8]   
green = df.iloc[:, 14]  
red = df.iloc[:, 26]    
blue = blue.to_numpy()
green = green.to_numpy()
red = red.to_numpy()
print("len(green): ", len(green))
end_pandas  = time.perf_counter()

print('Time to run pandas version: ', end_pandas-start_pandas, ' seconds')

#import timeit
#import numpy as np

#setup_code = """
#nums = list(range(100000))

# list_comp_time = timeit.timeit("[x**2 for x in nums]", setup=setup_code, number=100)
# map_time = timeit.timeit("list(map(lambda x: x**2, nums))", setup=setup_code, number=100)

# print("List comprehension time: ","%.5f"%list_comp_time ," seconds")
# print("Map function time: ",      "%.5f"%map_time       ," seconds")
# print("")

# setup_code = "nums_list = list(range(100000)); nums_set = set(nums_list)"
# list_time = timeit.timeit("99999 in nums_list", setup=setup_code, number=10000)
# set_time = timeit.timeit("99999 in nums_set", setup=setup_code, number=10000)

# print(f"List membership test time: ", "%.5f"%list_time ," seconds")
# print(f"Set membership test time: ",  "%.5f"%set_time,  " seconds")
# print("")

# setup_code = "import numpy as np; my_array = np.arange(100000)"

# sum([x**2 for x in range(100000)])
# np.sum(my_array**2)

# loop_time = timeit.timeit("", setup=..., number=...)
# numpy_time = timeit.timeit("", ...)

