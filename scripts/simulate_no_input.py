import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import cart_double_pendulum_model
from plotting import animate_state, cart_H, cart_W, initialize_objects

# Time setup
dt = 0.01
t_max = 45 # 45 seconds
T = np.arange(0, t_max, dt)

# Initial state 
# X in linear units
# theta1 in radians
# theta2 in radians
X0 = np.array([0, 0.3, 0.5, 0, 0, 0]).reshape(6)

# Solve DE
y = ode.odeint(lambda y,t : cart_double_pendulum_model(y, 0), X0, T)

# Setting up plot
fig = plt.figure()
ax = fig.add_subplot(autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
ax.set_aspect('equal')
ax.grid()

rect, line = initialize_objects(X0, ax)

ani = animation.FuncAnimation(fig, lambda i:animate_state(y, i, rect, line), len(y), interval=dt, blit=True)
plt.show()
