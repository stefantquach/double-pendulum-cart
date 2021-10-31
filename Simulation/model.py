import numpy as np
from numpy import cos, sin

# Base Constants
m =  1
m1 = 1
m2 = 1

L1 = 5
L2 = 5

g = 9.8

beta = 0.5
beta1 = 0.2
beta2 = 0.2

# Derived Constants
q1 = m + m1 + m2
q2 = (1/3*m1+m2)*L1*L1
q3 = 1/3*m2*L2*L2
q4 = (1/2*m1+m2)*L1
q5 = 1/2*m2*L2
q6 = 1/2*m2*L1*L2

##########################################################################################################
# written as D(theta)*theta_dotdot + C(theta, theta_dot)*theta_dot + G(theta) = H*u
# 
# Constants: m, m1, m2, L1, L2, g, beta, beta1, beta2
# Letting:
#    q1 = m + m1 + m2       q3 = 1/3*m2*L2^2    q5 = 1/2*m2*L2
#    q2 = (1/3*m1+m2)L1     q4 = (1/2*m1+m2)L1  q6 = 1/2*m2*L1*L2
# 
# D(theta) = [ q1                 q4*cos(theta1)           q5*cos(theta2)        ]
#            [ q4*cos(theta1)     q2                       q6*cos(theta1-theta2) ]
#            [ q5*cos(theta2)     q6*cos(theta1-theta2)    q3                    ]
#
# C(theta, theta_dot) = [ beta   -q4*theta1_dot*sin(theta_1)         -q5*theta2_dot*sin(theta_2)       ]
#                       [ 0      beta1                               q6*theta2_dot*sin(theta_1-theta2) ]
#                       [ 0      -q6*theta1_dot*sin(theta1-theta2)   beta2                             ]
# 
# G(theta) = [        0          ]
#            [ -g*q4*sin(theta1) ]
#            [ -g*q5*sin(theta2) ]
#
# H = [1]
#     [0]
#     [0]
##########################################################################################################
# Function to represent first order model
# X: state vector. expected to be [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
# u: input force
def cart_double_pendulum_model(X, u):
    X_prime = np.zeros(6)
    # print(X)
    # x = X[0]
    theta1 = X[1]
    theta2 = X[2]
    x_dot = X[3]
    theta1_dot = X[4]
    theta2_dot = X[5]

    # Theta = np.matrix([x, theta1, theta2]).reshape(3,1)
    Theta_dot = np.matrix([x_dot, theta1_dot, theta2_dot]).reshape(3,1)

    D = np.array( [[ q1,                 q4*cos(theta1),           q5*cos(theta2)        ],
                    [ q4*cos(theta1),     q2,                       q6*cos(theta1-theta2) ],
                    [ q5*cos(theta2),     q6*cos(theta1-theta2),    q3                    ]], dtype=np.float64)
    Dinv = np.linalg.inv(D) #  only really care about inv(D)
    # print(D)
    # print(Dinv)

    C = np.array([[ beta,      -q4*theta1_dot*sin(theta1),         -q5*theta2_dot*sin(theta2)         ],
                   [ 0,       beta1,                                   q6*theta2_dot*sin(theta1-theta2) ],
                   [ 0,      -q6*theta1_dot*sin(theta1-theta2),   beta2                                 ]], dtype=np.float64)

    G = np.array([[        0          ],
                   [ -g*q4*sin(theta1)  ],
                   [ -g*q5*sin(theta2) ]], dtype=np.float64)

    H = np.array([[u],[0],[0]], dtype=np.float64)

    F = np.array([[-beta],[beta1],[beta2]], dtype=np.float64)

    dd = Dinv@H - Dinv@C@Theta_dot - Dinv@G

    # Calculating derivative
    X_prime[0] = x_dot
    X_prime[1] = theta1_dot
    X_prime[2] = theta2_dot
    
    X_prime[3] = dd[0]
    X_prime[4] = dd[1]
    X_prime[5] = dd[2]
    return X_prime

