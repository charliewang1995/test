#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:39:04 2022

@author: ChaoruiWang
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


'''
The following function gives the right hand side of Lotke-Volterra in np.ndarray type
inputs:
    y: 2d vector
    alpha: real number
    beta: real number
    delta: real number
    gamma: real number
    
outputs:
    2d numpy array
'''
def lotke_volterra_rhs(y, alpha, beta, delta, gamma):
    return np.array([
                    alpha * y[0] - beta * y[0] * y[1], 
                    delta * y[0] * y[1] - gamma * y[1]   
                    ])


'''
The following function constructs the forward Euler method for ODEs
inputs:
    rhs_fun: function, describes RHS of ODEs, outputs 2d array
    T: positive real number, ending time
    N: positive integer, step numbers
    y0: 2d real vector, initial value
    alpha: real number
    beta: real number
    delta: real number
    gamma: real number
    
outputs:
    3d tuple
    y: 2d vector, approximate solution at time T
    x_traj: trajectory of first element of y along discretized time steps
    y_traj: trajectory of second element of y along discretized time steps
'''
def euler(rhs_fun, T, N, y0, alpha, beta, delta, gamma):
    if T <= 0:
        raise ValueError('Final time T should be positive.')
        
    if N<= 0:
        raise ValueError('N should be a positive integer.')
        
    try:
        if N == int(N):
            N = int(N)
    except TypeError:
        print('N should be an integer.')
        
    # compute Delta t
    dt = T/N
    
    x_traj = [y0[0]]
    y_traj = [y0[1]]
    
    # force y0 to float array
    y = np.array(y0) + np.zeros(2)
    
    for i in range(N):
        # use forward euler formula to update y at each timestep
        y += dt * rhs_fun(y, alpha, beta, delta, gamma)
        x_traj.append(y[0])
        y_traj.append(y[1])
    return (y, x_traj, y_traj)


'''
We use our euler function to solve the following ivp of Lotke-Volterra, where
T =1.0, N =1000, initial value = [10,20], all coefficients = 0.1
'''
sol_euler = euler(lotke_volterra_rhs, 1.0, 1000, [10,20], 0.1, 0.1, 0.1, 0.1)
print('----------------------------------------------------------------------')
print('Euler method solution:', sol_euler[0])
 

# plotting the euler method solution along the time steps
t = np.linspace(0,1,1001)
plt.title('Euler')
plt.plot(t, sol_euler[1], label='x')
plt.plot(t, sol_euler[2], label='y')
plt.ylim([0,30])
plt.legend()
plt.show()



'''
Following function construct the two-step Adams-Bashforth method
inputs:
    rhs_fun: function, describes RHS of ODEs, outputs 2d array
    T: positive real number, ending time
    N: integer greater than 1, step numbers
    y0: 2d real vector, initial value
    alpha: real number
    beta: real number
    delta: real number
    gamma: real number
    
outputs:
    3d tuple
    y_2: 2d vector, approximate solution at time T
    x_traj: trajectory of first element of y along discretized time steps
    y_traj: trajectory of second element of y along discretized time steps
'''
def lin_2step(rhs_fun, T, N, y0, alpha, beta, gamma, delta):
    if T <= 0:
        raise ValueError('Final time T should be positive.')
        
    if N<= 1:
        raise ValueError('N should be an integer greater than 1.')
        
    try:
        if N == int(N):
            N = int(N)
    except TypeError:
        print('N should be an integer.')
        
    dt = T/N
    y_0 = np.array(y0) + np.zeros(2)
    
    # just use Euler to create a necessary first step 
    y_1 = y_0 + dt * rhs_fun(y_0, alpha, beta, gamma, delta)
    x_traj = [y_0[0], y_1[0]]
    y_traj = [y_0[1], y_1[1]]
    for i in range(N-1):
        y_2 = y_1 + 3/2 * dt * rhs_fun(y_1, alpha, beta, gamma, delta) - 1/2 * dt * rhs_fun(y_0, alpha, beta, gamma, delta)
        x_traj.append(y_2[0])
        y_traj.append(y_2[1])
        y_0 = y_1
        y_1 = y_2
    return (y_2, x_traj, y_traj)


'''
Try the same problem as the Euler method, with all same parameters.
'''
sol_lin_2step = lin_2step(lotke_volterra_rhs, 1.0, 1000, [10,20], 0.1, 0.1, 0.1, 0.1)
print('----------------------------------------------------------------------')
print('Linear two-step solution:', sol_lin_2step[0])


plt.title('Linear Two-Step')
plt.plot(t, sol_lin_2step[1], label='x')
plt.plot(t, sol_lin_2step[2], label='y')
plt.ylim([0,30])
plt.legend()
plt.show()


'''
Let's do a test using dz/dt = -z. 
The exact solution should be e^-1, with T = 1.0, initial value = 1.0.
'''
sol_euler_test = euler(lambda y,alpha,beta,delta,gamma : -y, 1.0, 1000, [1, 0], -1.0, 0, 0, 0)
print('--------------------------------test-----------------------------------')
print('Exact solution:', np.exp(-1))
print('Solution by Euler:', sol_euler_test[0][0])
plt.title('Euler for dz/dt=-z')
plt.plot(t,sol_euler_test[1])
plt.legend()
plt.show()


'''
Let's do a test using simple harmonic oscillator. 
The exact solution should be
'''
sol_harmo_osci = euler(lambda y,alpha,beta,delta,gamma : -y, 1.0, 1000, 1, 0, 1, 0 , -0.1)
print(np.cos)


# use existing library introduced in the lecture, in particular the 'solve_ivp' function
def lv_rhs(t, y, alpha, beta, gamma, delta):
    return np.array([
                     alpha*y[0] - beta*y[0]*y[1],
                      delta*y[0]*y[1] - gamma*y[1]
                    ])

sol_ivp = solve_ivp(lambda t,y : lv_rhs(t ,y, 0.1, 0.1, 0.1, 0.1), [0,1], [10,20])
print(sol_ivp)
