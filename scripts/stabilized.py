import numpy as np
import scipy.integrate as ode
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import cart_double_pendulum_model
from plotting import animate_state, cart_H, cart_W, initialize_objects
from pole_placement import linearize

linearization_point = [0,0,0,0,0,0]
A,B = linearize(linearization_point)

# Calculating gains of controller
poles = [-20, -0.5, -0.51, -1, -6, -9]
K_obj = place_poles(A, B, poles)
K = K_obj.gain_matrix
print("inputted poles: ", K_obj.requested_poles)
print("Computed poles: ", K_obj.computed_poles)
print("gain matrix: ", K)

# defining model with input defined
def stabilized_model(X, t):
    return cart_double_pendulum_model(X, -np.dot(K,X))


# Time setup
dt = 0.01
t_max = 30 # 45 seconds
T = np.arange(0, t_max, dt)

# Initial state 
# X in linear units
# theta1 in radians
# theta2 in radians
X0 = np.array([0, 0.2, 0, 0, 0, 0]).reshape(6)

# Solve DE
y = ode.odeint(stabilized_model, X0, T)

# Setting up plot
fig = plt.figure()
ax = fig.add_subplot(autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
ax.set_aspect('equal')
ax.grid()

rect, line = initialize_objects(X0, ax)

ani = animation.FuncAnimation(fig, lambda i:animate_state(y, i, rect, line), len(y), interval=dt, blit=True)
plt.show()
