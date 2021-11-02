import sympy as sym
from sympy import cos, sin, pprint

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

## Declaring symbols
# Constants
q1 = sym.Symbol('q1')
q2 = sym.Symbol('q2')
q3 = sym.Symbol('q3')
q4 = sym.Symbol('q4')
q5 = sym.Symbol('q5')
q6 = sym.Symbol('q6')
g = sym.Symbol('g')
beta = sym.Symbol('beta')
beta1 = sym.Symbol('beta1')
beta2 = sym.Symbol('beta2')

# variables
x = sym.Symbol('x')
x_dot = sym.Symbol('x_dot')
theta1 = sym.Symbol('theta1')
theta1_dot = sym.Symbol('theta1_dot')
theta2 = sym.Symbol('theta2')
theta2_dot = sym.Symbol('theta2_dot')
u = sym.Symbol('u')

# Forming matricies
D = sym.Matrix([[ q1,                 q4*cos(theta1),           q5*cos(theta2)        ],
                [ q4*cos(theta1),     q2,                       q6*cos(theta1-theta2) ],
                [ q5*cos(theta2),     q6*cos(theta1-theta2),    q3                    ]])

C = sym.Matrix([[ beta,   -q4*theta1_dot*sin(theta1),         -q5*theta2_dot*sin(theta2)       ],
                [ 0,      beta1,                              q6*theta2_dot*sin(theta1-theta2) ],
                [ 0,      -q6*theta1_dot*sin(theta1-theta2),  beta2                            ]])

G = sym.Matrix([[        0          ],
                [ -g*q4*sin(theta1) ],
                [ -g*q5*sin(theta2) ]])

H = sym.Matrix([[1],[0],[0]])

Dinverse = D.inv()*C
sym.init_printing(use_unicode=False)
pprint(Dinverse)