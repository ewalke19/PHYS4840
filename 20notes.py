import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Gravity (m/s^2)
l = 0.40   # Length of pendulum arms (m)
m = 1   # Mass of pendulums (kg)

# Initial conditions
theta1 = np.radians(90)
theta2 = np.radians(90)
omega1 = 0.0
omega2 = 0.0
# State vector r = [theta1, theta2, omega1, omega2]
r0 = np.array([theta1, theta2, omega1, omega2])  

# Time parameters
dt = 0.01  # Time step
t_max = 10  # Simulation duration: sets number of TIME STEPS
t = np.arange(0, t_max, dt)

# Equations of motion for the double pendulum
def equations(r):
    ## assign the four variables we need to evolve to ONE vector r 
    ## that holds them all
    theta1, theta2, omega1, omega2 = r
    delta_theta = theta2 - theta1

    # Define the four equations for the system
    ftheta1 = omega1
    ftheta2 = omega2

    ## HINT: the expressions for fomega1, fomega2 are quite long,
    ## so create smaller expressions to hold the denominators
    denom1 = (2 * m * l ** 2)
    denom2 = (m * l ** 2)

    fomega1 = (-g * (2 * m) * np.sin(theta1) - m * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(delta_theta) * m *(omega2 ** 2 * l + omega1 ** 2 * l * np.cos(delta_theta))) / denom1

    fomega2 = (2 * np.sin(delta_theta) * (omega1 ** 2 * l * m + g * m * np.cos(theta1) + omega2 ** 2 * l * m * np.cos(delta_theta))) / denom2

    return np.array([ftheta1, ftheta2, ...])

# Runge-Kutta 4th order method
def rk4_step(r, dt):
    k1 = dt * equations(r)
    k2 = dt * equations(r + 0.5*k1)
    k3 = dt * equations(r + 0.5*k2)
    k4 = dt * equations(r + k3)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

## this is a CLEVER way to hold all of your data in one object
## R is a vector of lenght t (time steps) that will hold the evolution
## of all FOUR of your variables
## r0 is a VECTOR initialized to r0 = [0,0,0,0]
R = np.zeros((len(t), 4))
R[0] = r0

# Integrate equations and save data
## remember: numerical integration --> for loop
for i in range(1, len(t)):
    R[i] = rk4_step(R[i - 1], dt)

# Extract angles and angular velocities
theta1_vals, theta2_vals, omega1_vals, omega2_vals = R.T

# Convert to Cartesian coordinates for visualization
x1 = l * np.sin(theta1_vals)
y1 = -l * np.cos(theta1_vals)
x2 = x1 + l * np.sin(theta2_vals)
y2 = y1 - l * np.cos(theta2_vals)

# Save data
np.savetxt("double_pendulum_data.txt", np.column_stack([t, x1, y1, x2, y2]),
           header="time x1 y1 x2 y2", comments="")


data = np.loadtxt("double_pendulum_data.txt", skiprows=1)
t, x1, y1, x2, y2 = data.T

plt.figure(figsize=(6, 6))
plt.plot(x1, y1, marker='.', label="Mass 1 (Path)")
plt.plot(x2, y2, marker='.', label="Mass 2 (Path)", color="red")
plt.scatter([0], [0], color="black", marker="o", label="Pivot")

plt.scatter([x1[0]], [y1[0]], color="blue", marker="+", s=100, label="Mass 1 (Start)", zorder=3)
plt.scatter([x2[0]], [y2[0]], color="red", marker="+", s=100, label="Mass 2 (Start)", zorder=3)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Double Pendulum Motion")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()


# video 
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_title("Double Pendulum Simulation")

pivot, = ax.plot([], [], 'ko', label="Pivot")

line1, = ax.plot([], [], 'b-', label="Mass 1 Path")
line2, = ax.plot([], [], 'r-', label="Mass 2 Path")


mass1, = ax.plot([], [], 'bo', label="Mass 1", markersize=8)
mass2, = ax.plot([], [], 'ro', label="Mass 2", markersize=8)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    mass1.set_data([], [])
    mass2.set_data([], [])
    return line1, line2, mass1, mass2
def update(frame):
    # Get the current positions of the masses
    x1_pos = x1[frame]
    y1_pos = y1[frame]
    x2_pos = x2[frame]
    y2_pos = y2[frame]
    
    # Update the data for the lines
    line1.set_data([0, x1_pos], [0, y1_pos])  # Line from pivot to mass 1
    line2.set_data([x1_pos, x2_pos], [y1_pos, y2_pos])  # Line from mass 1 to mass 2

    # Update the positions of the masses
    mass1.set_data(x1_pos, y1_pos)
    mass2.set_data(x2_pos, y2_pos)
    
    return line1, line2, mass1, mass2

# Set up the animation
# Adjust interval and fps
interval_ms = 10  # 200 ms between frames
fps = 1000 // interval_ms  # Ensure the fps matches the interval

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval_ms)

# Save the animation as a video (MP4 file)
#ani.save('double_pendulum_simulation.mp4', writer='ffmpeg', fps=fps)

plt.show()




