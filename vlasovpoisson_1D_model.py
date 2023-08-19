#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannahhaider
"""

#%%
# import needed packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import matplotlib
from scipy.sparse import diags
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
import math
from random import sample 
import imageio
import os 

from vpmodelfunctions import *

#%% 
# set plot fontsize
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#%%

"""                         BUILDING THE MESH

- 1-dimensional, 1-velocity scheme which will approximate both position and 
velocity of Np particles. Thus, the state vector is comprised of 2 states: 
position and velocity, denoted as capital X and V.

These states are initialized by considering the initial probability distribution 
function f_sol = f(x,v,t = 0). 

spatial grid defined by x vector, with Nx spatial steps.
velocity grid defined by v vector, with Nv velocity steps.
time evolution defined by t vector, with Nt time steps. 

This scheme requires that Nx = Np for symplectic time integration to accurately be implemented.
"""
#%% constants 

# defining mesh indices
Nx = 200 # spatial grid points
Nv = 300 # velocity grid points
Nt = 2000 # time steps

# defining PIC parameters
Np = Nx # number of macroparticles, must equal number of spatial grid points Nx
w = 1/Np # weight of each particle
q_m_ratio = -1 # elementary charge to mass ratio

#%% defining meshgrid

# defining position vector
xinitial = 0 # initial position
xfinal = 4*np.pi # final position

x = np.linspace(xinitial, xfinal, Nx) # position vector, x
dx = x[1] - x[0]

# defining time vector
tinitial = 0  # initial time
tfinal = 40 # final time

t = np.linspace(tinitial, tfinal, Nt + 1)
dt = t[1]- t[0] # time step

#defining velocity (momentum) vector
vinitial = -4
vfinal = 4 # these are arbitrarily assigned and not informed by latter
# research papers, and might affect the uncertainty of our model.

v = np.linspace(vinitial, vfinal, Nv) # momentum vector, v
dv = v[2] - v[1]

#%% 
"""PLOTTING INITIAL CONDITIONS 

one may choose either the first initial condition given by function 

'initial_distribution_function' or the second, given by the function 

'landau_damping_IC'. Here, the second is modeled. 
"""

# initialize distribution function
f_sol = landau_damping_IC(x = x, v = v, Nx = Nx, Nv = Nv, Nt = Nt)

fig, ax = plt.subplots(nrows = 1)
pos = ax.pcolormesh(x, v, f_sol[0,:,:].T)
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(r"$f (x, v, t =0)$")
fig.colorbar(pos, ax = ax)
# Save figure
filename = 'initial_condition.png'
plt.savefig(filename, dpi = 360)

#%% 
""" REJECTION SAMPLING FOR THE INITIAL CONDITIONS """

# this envelope is an easier distribution, a random uniform distribution of particles 
envelope = np.random.uniform(low = np.min(f_sol[0, :, :]), high = np.max(f_sol[0, :, :]), size = f_sol[0, :, :].shape)

# continue rejection sampling from sampling function 

samples = [] # list of samples
number_samples_accepted = 0 # initial number of samples 
index = (envelope <= f_sol[0, :, :]) # returns true at grid point where sampling
# function is in range of initial distribution function

while number_samples_accepted < 2000:
    
    for ii in range(Nx):
        for jj in range(Nv):
        
            if index[ii, jj] == True:
                samples.append(( x[ii], v[jj] ))
        
                number_samples_accepted += 1
            
#               
# now we have our samples 
samples = np.array(samples) # change list object to array 

# allocate samples to initial distribution 
X = samples[:, 0]
V = samples[:, 1]

# now only gathering Np samples
X = np.random.choice(X, size = Np, replace = False) # changing the replace option to True should only be done if num_samples < Np
V = np.random.choice(V, size = Np, replace = False)

# compute actual initial distribution based on subsampled X and V

initial_solution = update_distribution_function(Nx = Nx, 
                                                Nv = Nv, 
                                                X = X, 
                                                V = V, 
                                                x = x, 
                                                v = v, 
                                                dx = dx)
#%%
""" plot ENVELOPE aka sampling function & INITIAL PARTICLE DISTRIBUTION of N_p
particles """  

# plot envelope 

fig, ax = plt.subplots(nrows = 1)
pos = ax.pcolormesh(x, v, envelope.T)
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(r"Envelope at time t = 0")
fig.colorbar(pos, ax = ax)
# Save figure
filename = 'sampling_function.png'
plt.savefig(filename, dpi = 360)


# plot initial particle distribution 

fig, ax = plt.subplots(nrows = 1)
pos = ax.pcolormesh(x, v, initial_solution.T)
ax.set_xlabel("x")
ax.set_ylabel("v")
ax.set_title(r"particles at time t = 0")
fig.colorbar(pos, ax = ax)
# Save figure
filename = 'initialparticles.png'
plt.savefig(filename, dpi = 360)

# reallocating initial distribution 
f_sol[0, :, :] = initial_solution 


#%% initialize the time stepping algorithm by initializing charge density, 
# electric potential, electric field, and the Hamiltonian

# initialize rho using the initial, subsampled particle distribution 
rho = np.zeros(( len(t), len(X) )) 
rho[0, :] =  electric_charge_density(initial_solution,
                                Nv = Nv,
                                Nx = Nx)

# initialize electric potential, phi
phi = np.zeros(( len(t), len(X) ))
phi[0,:] = electric_potential(rho = rho[0,:], 
                              dx = dx, 
                              periodic = True)

# initialize electric field
E = np.zeros(( len(t), len(X) ))
E[0,:] = electric_field(phi = phi[0, :], 
                        dx = dx, 
                        periodic = True)

# compute initial Hamiltonian
initial_Hamiltonian = Hamiltonian(V = V, 
                                  phi = phi[0,:])

# initialize Hamiltonian to save over time 
Hamiltonian_evolution = np.zeros(( len(t) ))
Hamiltonian_evolution[0] = initial_Hamiltonian 

# initialize separable Hamiltonian over time 
Hamiltonian_X = np.zeros(( len(t) ))
Hamiltonian_X[0] = Hamiltonian_x(phi = phi[0, :], q_m_ratio = q_m_ratio)
Hamiltonian_V = np.zeros(( len(t) ))
Hamiltonian_V[0] = Hamiltonian_v(V = V)

# begin animation process by creating lists for the filenames 
filenames_sol = [] # saving the filenames for the distribution function at different time steps
filenames_parameters = [] # saving the filenames for the charge density, electric 
# potential, and electric field at different time steps 

#%% 
""" BEGIN SYMPLECTIC TIME INTEGRATION """

for ii in range(Nt):

    # first symplectic step, half step forward in Velocity V
    V_half = V - ( 0.5 * dt * q_m_ratio * - E[ii, :] )
    
    # second symplectic step, full step forward in X 
    X = X + ( dt * V_half )
    
    # enforce spatially periodic boundary conditions
    X = boundary_conditions(X = X, x = x)
    
    # update distribution function with X^(n + 1) and V^(n + 1/2)
    f_sol_pre = update_distribution_function(Nx = Nx, 
                                   Nv = Nv, 
                                   X = X, 
                                   V = V_half, 
                                   x = x,
                                   v = v, 
                                   dx = dx)
    
    # evaluate electric charge density at X^(n+1) and V^(n + 1/2)
    rho_pre = electric_charge_density(f = f_sol_pre,
                                  Nv = Nv,
                                  Nx = Nx)

    # evaluate electric potential at X_n+1 and V_n+1/2, can be interchanged w GMRES
    # function
    phi_pre = electric_potential(rho = rho_pre, 
                                 dx = dx, 
                                 periodic = True)

   
    # evaluate electric field at X^(n + 1) and V^(n + 1/2)
    E_pre = electric_field(phi = phi_pre, 
                           dx = dx, 
                           periodic = True)
   
    
    # third symplectic step, full step forward to V^(n + 1)
    V = V - (0.5 * dt * q_m_ratio * - E_pre)

    
    # update distribution function for the next step in time (n+1)
    f_sol[ii + 1, :, :] = update_distribution_function(Nx = Nx, 
                                             Nv = Nv, 
                                             X = X, 
                                             V = V, 
                                             x = x, 
                                             v = v,
                                             dx = dx)
 
    
    # evaluate electric charge density for the next step in time
    rho[ii + 1, :] = electric_charge_density(f = f_sol[ii+1, :, :],
                                  Nv = Nv,
                                  Nx = Nx)

    # evaluate electric potential for the next step in time, can be interchanged w GMRES
    # function
    phi[ii + 1, :] = electric_potential(rho = rho[ii + 1, :], 
                                        dx = dx, 
                                        periodic = True)
    
    # evaluate electric field for the next step in time
    E[ii + 1, :] = electric_field(phi = phi[ii + 1, :], 
                                  dx = dx, 
                                  periodic = True)
    
    # evaluate Hamiltonian for the next step in time 
    Hamiltonian_evolution[ii + 1] = Hamiltonian(V = V, 
                                                phi = phi[ii +1, :])
    Hamiltonian_X[ii + 1] = Hamiltonian_x(phi = phi[ii + 1],
                                          q_m_ratio = q_m_ratio)
    Hamiltonian_V[ii + 1] = Hamiltonian_v(V = V)
    
    print(ii)
    
    
    if ii %10 == 0:
        
        # save plots for the distribution function at different time steps 
        fig4, ax = plt.subplots()
        pos = ax.pcolormesh(x, v, f_sol[ii + 1,:,:].T)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(r"$f (x, v, t = $" + str(round(t[ii + 1], 3)) + r"$)$")
        fig.colorbar(pos, ax = ax)
        plt.tight_layout()
        
        filename = f'solution{ii + 1}.png' # create filename 
        filenames_sol.append(filename)
        fig4.savefig(filename)
        plt.close()
        
        # save plots for the flow parameters at different time steps 
        fig5, ax = plt.subplots(nrows = 3)
        pos = ax[0].plot(x, -E[ii + 1,:], "m")
        ax[0].set_ylabel("E")
        ax[0].set_title(r"$E(x), \phi(x), \rho(x)$")
        pos = ax[1].plot(x, phi[ii + 1, :], "b")
        ax[1].set_ylabel(r"$\phi$")
        pos = ax[2].plot(x, rho[ii + 1,:], "c")
        ax[2].set_xlabel("x")
        ax[2].set_ylabel(r"$\rho$")
        fig5.suptitle("Flow parameters at t = " + str(round(t[ii + 1], 3)))
        plt.tight_layout()
        
        filename = f'flowparam{ii + 1}.png'
        filenames_parameters.append(filename)
        fig5.savefig(filename)
        plt.close()
    
#%% Check Hamiltonian relative error 
Final_Hamiltonian = Hamiltonian(V = V, phi = phi[-1, :])

error_tol = 1e-08

if np.abs(((Final_Hamiltonian - initial_Hamiltonian)) / initial_Hamiltonian) > error_tol:
    print("Hamiltonian is not conserved :-(")
else:
    print('Hamiltonian is conserved!!! :)))')

relative_error = (np.abs((Final_Hamiltonian - initial_Hamiltonian) / initial_Hamiltonian ))*100
#%%  
""" ANIMATE DISTRIBUTION FUNCTION AND FLOW PARAMETERS THROUGH TIME """

# build gif for distribution function
with imageio.get_writer('solution.gif', mode = 'I') as writer:
    for filename in filenames_sol:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames_sol):
    os.remove(filename)  


# build gif for flow parameters 
with imageio.get_writer('flow_param_solution.gif', mode = 'I') as writer:
    for filename in filenames_parameters:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames_parameters):
    os.remove(filename)  
        

#%%
""" PLOT INITIAL CONDITIONS FOR RHO, PHI, AND E """

fig3, ax = plt.subplots(nrows = 3, figsize = (5, 8))
pos = ax[0].plot(x, -E[0,:], "m")
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$E(x, t = 0)$", fontsize = 10)
pos = ax[1].plot(x, phi[0, :], "b")
ax[1].set_xlabel("x")
ax[1].set_ylabel(r"$\phi(x, t = 0)$", fontsize = 10)
pos = ax[2].plot(x, rho[0,:], "c")
ax[2].set_xlabel("x")
ax[2].set_ylabel(r"$\rho(x, t = 0)$", fontsize = 10)
fig3.suptitle("Flow parameters at t = 0")
filename = "icparameters.png"

plt.tight_layout()
plt.savefig(filename, dpi = 360)


#%% 

""" PLOT HAMILTONIAN EVOLUTION THROUGH TIME"""

fig4, ax = plt.subplots(figsize = (7,5))
plt.plot(t, Hamiltonian_evolution, "b")
plt.xlabel("time (s)")
plt.ylabel(r"$H(X,V) = \Sigma_{i=1}^{n} [\frac{1}{2} V_{i}^{2} + \frac{q}{m} \Phi(X_i)]$ ")
plt.title("Hamiltonian over time")
plt.grid()
plt.tight_layout()
plt.savefig('Hamiltonian_over_time.png')


#%% 

""" CALCULATE AND PLOT THEH HAMILTONIAN ERROR OVER TIME """
# initializing error data structure 
Hamiltonian_error = np.zeros((len(t)))
for i in range(len(t)):
    
    Hamiltonian_error[i] = np.abs(Hamiltonian_evolution[i] - initial_Hamiltonian)

fig5, ax = plt.subplots(figsize = (7,5))
plt.plot(t, Hamiltonian_error, "b")
plt.xlabel("time (s)")
plt.ylabel(r"$H_d(\mathbf{u}(t)) - H_d(\mathbf{u}(0))$ ")
plt.title("Absolute error over time ")
plt.grid()
plt.tight_layout()
plt.savefig('absolute_error.png')

#%% 
""" PLOT SEPARABLE HAMILTONIAN IN X """

fig6, ax = plt.subplots(figsize = (7,5))
plt.plot(t, Hamiltonian_X, "b")
plt.xlabel("time (s)")
plt.ylabel(r"$H_X = \Sigma_{i=1}^{n}\frac{q}{m} \Phi(X)$")
plt.title(r"$H_X$")
plt.grid()
plt.tight_layout()
plt.savefig('separable_Ham_X.png')
#%% 
""" PLOT SEPARABLE HAMILTONIAN IN X """

fig6, ax = plt.subplots(figsize = (7,5))
plt.plot(t, Hamiltonian_V, "b")
plt.xlabel("time (s)")
plt.ylabel(r"$H_V = \Sigma_{i=1}^{n} \frac{1}{2} V_{i}^{2}$")
plt.title(r"$H_V$")
plt.grid()
plt.tight_layout()
plt.savefig('separable_Ham.png')