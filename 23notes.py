from numpy import empty,zeros,max
from pylab import imshow,gray,show

# Constants
M = 100         # Grid squares on a side
h = 1           # length between adjacent nodes

V = 1.0         # Voltage at top wall
target = 1e-6   # Target accuracy -- tolerance threshold for solution

# Create arrays to hold potential values
phi = zeros([M+1,M+1],float) ## 2D array 


'''
the following statement is setting a Dirichlet boundary condition on the top edge of the 2D grid
phi is a 2D NumPy array of shape (M+1, M+1) representing the potential at each point on a square grid
The notation phi[0,:] means: “all columns in row 0” — in other words, the entire top row of the grid
phi[0,:] = V sets the potential to V = 1.0 on the entire top boundary.
All other boundaries (bottom, left, and right) are implicitly left at zero 
(since phi was initialized with zeros(...)), meaning those edges are held at 0 volts.
'''
phi[0,:] = V    


phinew = empty([M+1,M+1],float)

# Main loop
delta = 1.0
while delta>target:

    # Calculate new values of the potential
    for i in range(M+1):
        for j in range(M+1):
            ## boundary conditions
            if i==0 or i==M or j==0 or j==M:
                phinew[i,j] = phi[i,j]
            else:
                phinew[i,j] = (phi[i+h,j] + phi[i-h,j] \
                                 + phi[i,j+h] + phi[i,j-h])/4.

    # Calculate maximum difference from old values
    delta = max(abs(phi-phinew))

    phi = phinew  # the new value of phi is set to what we just found for phinew
    phinew = phi  # phinew will be immediately overwritten in the next iteration, so 
                  # we assign it a placeholder value of the correct size until then, 
                  # which might as well be phi


    # shorthand way of doing this is to simply swap the two arrays around
    #   phi,phinew = phinew,phi

# Make a plot
imshow(phi)
gray()
show()





import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 30              # Grid size (cube of size N x N x N)
h = 1               # Grid spacing
V = 1.0             # Voltage on the top face (z = 0)
target = 1e-7       # Convergence criterion

# Initialize the potential arrays
phi = np.zeros((N+1, N+1, N+1), dtype=float)
phinew = np.empty_like(phi)

# Apply boundary condition: top face (z = 0) at V, others at 0
phi[:,:,0] = V

# Iterative solution using Gauss-Seidel-like update
delta = 1.0
iteration = 0
while delta > target:
    iteration += 1
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                phinew[i,j,k] = (phi[i+h,j,k] + phi[i-h,j,k] +
                                 phi[i,j+h,k] + phi[i,j-h,k] +
                                 phi[i,j,k+h] + phi[i,j,k-h]) / 6.0

    # Preserve boundary conditions
    phinew[:,:,0] = V
    phinew[:,:,N] = 0
    phinew[:,0,:] = 0
    phinew[:,N,:] = 0
    phinew[0,:,:] = 0
    phinew[N,:,:] = 0

    delta = np.max(np.abs(phi - phinew))
    phi, phinew = phinew, phi

    if iteration % 10 == 0:
        print(f"Iteration {iteration}, max delta = {delta:.2e}")

# Visualization: middle slice in z-direction
mid_z = N // 2
plt.figure(figsize=(6,5))
plt.imshow(phi[:,:,mid_z], origin='lower', cmap='inferno')
plt.colorbar(label='Potential $\Phi$')
plt.title(f"Midplane slice at z = {mid_z}")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()


# a) Phinew has 6 terms instead of 4 because we now have to change k which adds
# the terms phi[i,j,k_h] and phi[i,j,k-h] which changes phinew. 

# b) The grid size is made smaller beacause it takes a few second to load the pixels. 
# If it was 100,100,100 is would take a while. Or it's printing the iteration that is 
# making it load slow. 

# c) Changing the convergence criterion changed the iteration and max delta. If 
# it is made smaller, (1e-10), iteration and max delta produce a lot more numbers. 

# d ) the boundary conditions in the 2D array are preserved in the y (j) direction
# and in the 3D array, they are preserved in the z (k) direction. 



# np.roll() takes values beyond the last position and introduces them back in the 
# beginning.







