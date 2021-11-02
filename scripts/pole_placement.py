import numpy as np
from numpy import cos, sin
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
#
# To Linearize G, need delG/delTheta
# delG/delTheta = [ 0       0                  0                ]
#                 [ 0       -g*q4*cos(theta1)  0                ]
#                 [ 0       0                  -g*q5*cos(theta2)]
# This matrix then gets absorbed into the A matrix
##########################################################################################################

# importing constants
from model import *

def linearize(X0):
    # Linearization point
    x = X0[0]
    theta1 = X0[1]
    theta2 = X0[2]
    x_dot = X0[3]
    theta1_dot = X0[4]
    theta2_dot = X0[5]

    # To linearize, we use D(0), C(0,0), del G/ del theta (0)
    D = np.array([[q1, q4, q5]], dtype=np.float64)

    # Theta = np.matrix([x, theta1, theta2]).reshape(3,1)
    # Theta_dot = np.matrix([x_dot, theta1_dot, theta2_dot]).reshape(3,1)

    D = np.array( [[ q1,                 q4*cos(theta1),           q5*cos(theta2)        ],
                    [ q4*cos(theta1),     q2,                       q6*cos(theta1-theta2) ],
                    [ q5*cos(theta2),     q6*cos(theta1-theta2),    q3                    ]], dtype=np.float64)
    Dinv = np.linalg.inv(D) #  only really care about inv(D)

    C = np.array([[ beta,      -q4*theta1_dot*sin(theta1),         -q5*theta2_dot*sin(theta2)         ],
                    [ 0,       beta1,                                   q6*theta2_dot*sin(theta1-theta2) ],
                    [ 0,      -q6*theta1_dot*sin(theta1-theta2),   beta2                                 ]], dtype=np.float64)

    # G = np.array([[        0          ],
    #                 [ -g*q4*sin(theta1)  ],
    #                 [ -g*q5*sin(theta2) ]], dtype=np.float64)
    delG = np.array([[ 0,       0,                  0                ],
                    [ 0,       -g*q4*cos(theta1),  0                ],
                    [ 0,       0,                  -g*q5*cos(theta2)]], dtype=np.float64)

    H = np.array([[1],[0],[0]], dtype=np.float64)


    # X_dot = AX + Bu 
    # Standard LTI System representation
    A = np.zeros((6,6))
    A[0:3, 3:6] = np.identity(3)
    A[3:6, 0:3] = -Dinv@delG
    A[3:6, 3:6] = Dinv@C

    B = np.zeros((6,1))
    B[3:6] = Dinv@H

    return A,B

