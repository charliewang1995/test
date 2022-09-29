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
    return delta*x*y - gamma*y

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


# linear multistep two-step Adams–Bashforth
def Llvodes_lin_multi(T, N, x_init, y_init, alpha, beta, gamma, delta):
    if N <= 1:
        raise ValueError('The step number should be bigger or equal than 2.')
    dt = T/N
    x0 = x_init
    y0 = y_init
    x1 = x0 + dt*ode1_rhs(x0, y0, alpha, beta)
    y1 = y0 + dt*ode2_rhs(x0, y0, gamma, delta)
    x_traj = [x_init, x_firststep]
    y_traj = [y_init, y_firststep]
    x = x_init
    x_firststep += dt*ode1_rhs(x, y, alpha, beta)
    y = y_init
    y_firststep = dt*ode2_rhs(x, y, gamma, delta)
    for i in range(N-1):
        x2 = x1 + 3/2*dt*ode1_rhs(x1, y1, alpha, beta) - 1/2*ode1_rhs(x0, y0, alpha, beta)
        y2 = y1 + 3/2*dt*ode2_rhs(x1, y1, gamma, delta) - 1/2*ode2_rhs(x0, y0, gamma, delta)
        x1 = x2
        y1 = y2
    return


# testing
sol_test = lv_odes_euler(1.0, 1000, 1, 0, -1.0, 0, 0, 0)
print(np.exp(-1))
print(sol_test[0])
plt.plot(t,sol_test[2])
plt.show()




# use existing library introduced in the lecture
def lv_rhs(t, y, alpha, beta, gamma, delta):
    return np.array([
                     alpha*y[0] - beta*y[0]*y[1],
                      delta*y[0]*y[1] - gamma*y[1]
                    ])

sol_ivp = solve_ivp(lambda t,y : lv_rhs(t ,y, 0.1, 0.1, 0.1, 0.1), [0,1], [10,20])
print(sol_ivp)
