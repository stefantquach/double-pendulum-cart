import matplotlib.pyplot as plt
import numpy as np
from model import L1, L2

cart_H = 1
cart_W = 2

def animate_state(X, i, rect, line):
    # Extracting necessary variables
    x_cart = X[i,0]
    theta1 = X[i,1]
    theta2 = X[i,2]

    # Drawing cart
    cart_llx = x_cart - cart_W/2
    cart_lly = -cart_H/2
    rect.set_xy([cart_llx, cart_lly])

    # Drawing pendulum
    x1 = x_cart + L1*np.sin(theta1)
    y1 = L1*np.cos(theta1)
    x2 = x_cart + L1*np.sin(theta1) + L2*np.sin(theta2)
    y2 = L1*np.cos(theta1) + L2*np.cos(theta2)

    line.set_data([x_cart, x1, x2], [0, y1, y2])
    return rect, line


def initialize_objects(X, ax):
    x_cart = X[0]
    theta1 = X[1]
    theta2 = X[2]

    rectangle = plt.Rectangle([x_cart-cart_W/2, -cart_H], cart_W, cart_H, fill=False)
    patch = ax.add_patch(rectangle)
    
    x1 = x_cart + L1/2*np.sin(theta1)
    y1 = L1/2*np.cos(theta1)
    x2 = x_cart + L1*np.sin(theta1) + L2/2*np.sin(theta2)
    y2 = L1*np.cos(theta1) + L2/2*np.cos(theta2)

    line, = plt.plot([x_cart, x1, x2], [0, y1, y2], '-o')

    return patch, line