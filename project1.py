#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:39:04 2022

@author: ChaoruiWang
"""
import numpy as np               # import numpy package
import matplotlib.pyplot as plt  # to plot
from scipy.integrate import solve_ivp


def ode1_rhs(x, y, alpha, beta):
    return alpha*x - beta*x*y

def ode2_rhs(x, y, gamma, delta):
    return delta*x - gamma*y

def forward_Euler(x, f, dt):
    return x+dt*f(x)

def lv_odes_euler(T, N, x_init, y_init, alpha, beta, gamma, delta):
    dt = T/N
    x_traj = [x_init]
    y_traj = [y_init]
    x = x_init
    y = y_init    
    for i in range(N):
        x += dt*ode1_rhs(x, y, alpha, beta)
        y += dt*ode2_rhs(x, y, gamma, delta) 
        x_traj.append(x)
        y_traj.append(y)
    return (x, y, x_traj, y_traj)

sol_euler = lv_odes_euler(1.0, 1000, 10, 20, 0.1, 0.1, 0.1, 0.1)
print(sol_euler[0],sol_euler[1])

# plotting solution x,y on the time interval
t = np.linspace(0,1,1001)
plt.plot(t, sol_euler[2], label='x')
plt.plot(t, sol_euler[3], label='y')
plt.ylim([0,30])
plt.legend()
plt.show()


# Runge Kutta
def lv_odes_RK4():
    
    return


# linear multistep
def Llvodes_lin_multi():
    
    return


# use existing library introduced in the lecture
def lv_rhs(t, y, alpha, beta, gamma, delta):
    return np.array([
                     alpha*y[0] - beta*y[0]*y[1],
                      delta*y[0] - gamma*y[1]
                    ])

sol_ivp = solve_ivp(lambda t,y : lv_rhs(t ,y, 0.1, 0.1, 0.1, 0.1), [0,1], [10,20])
print(sol_ivp)
