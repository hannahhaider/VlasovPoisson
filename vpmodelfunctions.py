#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hannahhaider

                    PHYSICS FUNCTIONS NEEDED FOR THE VLASOV POISSON MODEL

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


#%% functions to approximate the initial distribution function
def initial_distribution_function(x, v, Nx, Nv, Nt):
    """This function returns the initial distribution of particles for a bump-on-tail
    distribution in velocity and periodic spatial perturbation.
-------------------------------------------------------------------------------
    PARAMETERS
        inputs:
            x : Nx x 1 space vector
            v : Nv x 1 velocity vector
            Nx : spatial grid points
            Nv : velocity grid points
            Nt : time steps

        returns:
            f, initial distribution function.
-------------------------------------------------------------------------------
"""
    f = np.zeros((Nt + 1, Nx, Nv))

    
    epsilon = 0.3# arbitrarily assigned parameter in gaussian distribution for initial condition
    a = 0.3 # arbitrarily assigned parameter in gaussian distribution for initial condition
    v0 = 4 # arbitrarily assigned initial velocity parameter in gaussian distribution for initial condition
    sigma = .0071 # standard deviation
    
    # computing initial distribution function at time t=0
    for ii in range(Nx):
        for jj in range(Nv):
            term1 = (1 + epsilon*np.cos(2*np.pi*x[ii]))
            term2 = (1/ (1 + a))*( 1 / (np.sqrt(2*np.pi))) * np.exp(-0.5*v[jj]**2)
            term3 = (a/ ( 1 + a)) * (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp((-1/(2*sigma**2))*(v[jj] - v0)**2)
            f[0, ii, jj] = term1 * (term2 + term3)
            
    return f

def landau_damping_IC(x, v, Nx, Nv, Nt):
    """To model the weak damping or damped propogation of small apmplitude plasma waves, 
    we employ an initial condition given by a periodic spatial perturbation and velocity distribution function.
    
    Essentially, f(x, v, t = 0) = f(x, t= 0)*f(v, t = 0)
-------------------------------------------------------------------------------
    PARAMETERS
    
        inputs:
            x : Nx x 1 space vector
            v : Nv x 1 velocity vector
            Nx : spatial grid points
            Nv : velocity grid points
            Nt : time steps

        returns:
            f, initial distribution function.
-------------------------------------------------------------------------------
    """
    # initialize data structure
    f = np.zeros((Nt + 1, Nx, Nv))
    # constants 
    alpha = 0.4912# amplitude of perturbation, alpha chosen 
    k = 0.5 # wavenumber 
    sigma = .9889 # standard deviation of velocity Maxwellian 
    
    # computing initial distribution function at time t=0
    for ii in range(Nx):
        for jj in range(Nv):
            term1 = (1 + alpha*np.cos(k*x[ii]))
            term2 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp( - v[jj]**2 / (2*sigma**2))
            
            f[0, ii, jj] = term1 * term2

    return f


#%% updating distribution function 

def update_distribution_function(Nx, Nv, X, V, x, v, dx):
    """This function updates the distribution function for each time step. 
    
    Given the number of spatial and velocity steps, we may evaluate the Ansatz
    function to determine the particle distribution at a time t. 
-------------------------------------------------------------------------------
    Parameters:
        Nx = spatial steps, type = int
        Nv = velocity steps, type = int
        X = state vector of n particle positions 
        V = state vector of n particle velocities
        x = spatial grid points 
        v = velocity grid points 
        dx = spatial step, used as tolerance for filling grid points
        
    Returns:
        w * (dirac delta function evaluated for x dotted with dirac delta function
             evaluated for v)
-------------------------------------------------------------------------------
    """
    w = 1/(len(X))
    
    solution = np.zeros((Nx, Nv))
    
    for particle in range(len(X)):
        delta_X = np.zeros((Nx))
        delta_V = np.zeros((Nv))
        
        x_difference = np.abs(x - X[particle])
        v_difference = np.abs(v - V[particle])
        
        if np.any(x_difference <= dx) and np.any(v_difference <= dx):
                
            delta_X[x_difference <= dx] = 1
            delta_V[v_difference <= dx] = 1

        
        solution += w * np.outer(delta_X, delta_V)
        
                    
    return solution 

#%% spatial boundary condition enforcement 

def boundary_conditions(X, x):
    """This function bounds the right=handside and left-handside of the 
    distribution function by assuming a periodic spatial domain """
    
    # periodic spatial domain 
    
    return np.mod(X, x[-1])  # ensures X values are within domain of x
#%% poisson solver  

def electric_charge_density(f, Nv, Nx):
    """This function attempts to integrate the distribution function f (t, x, v)
    with respect to v in order to compute the spatial density at that time step.

    Essentially, given V at a certain time step, the spatial density is found by
    numerical integration.
-------------------------------------------------------------------------------
    PARAMETERS:
        inputs:
            f = distribution function, type = matrix w/ dimensions Nx x Nv
            Nv = velocity grid points, type = int
            Nx = spatial grid points, type = int

        Returns:
            rho(x) = spatially dependent electric charge density, type: vector
            of  with dimensions Nx x 1
            
-------------------------------------------------------------------------------
            """

    rho = np.zeros((Nx))

    for i in range(Nv): # for all velocity grid points

        rho += f[:, i] # for all rows, add each velocity grid point

    rho /= Nv

    return rho


def electric_potential(rho, dx, periodic = True):

    """ This function approximates the electric potential, phi, given an
    assumed spatially periodic, external electric field . In order to do this,
    we recall Poissons equation which states that the second derivative of phi
    with respect to x (spatial domain [0, L]) is equal to the - density (as
    a function of position and time). Thus, by employing a second order central
    difference of phi, with N + 1 cells, and assuming we have the empirical
    data for density, rho, one can obtain the global phi vector.

  -----------------------------------------------------------------------------

    PARAMETERS:
        inputs:
            rho(x) = 
            dx = 
            periodic = True/False 

        Returns:
            phi(x), type = vector with dimensions Nx x 1 

    ---------------------------------------------------------------------------
    Analytics:

    The exact numerical scheme is detailed as follows for periodic boundary
    conditions, ie phi(x) = phi(x+L):
        (phi_{i+1} - 2*phi_{i} + phi_{i-1})/(dx^2) = - rho_{i}.

    Thus, the matrix formulation A * phi = rho can be used to numerically
    determine the phi matrix. Here, A is a finite difference matrix"""

    A = diags([2, -1, -1], [0, -1, 1], shape = (rho.shape[0], rho.shape[0])).toarray()

    if periodic:
        A[-1, 0] = 1
        A[0, -1] = -1


    A /= (dx)**2

    # inversely solve the A @ phi == rho linear equation set for phi

    phi = np.linalg.solve(A, rho)

    return phi

def electric_potential_GMRES(rho, dx, periodic = True):
    """Similarly to the previous function, the Poisson equation is iteratively 
    solved using a generalized minimum residual error method, known as GMRES 
    in python's scipy package.
    -----------------------------------------------------------------------------

      PARAMETERS:
          inputs:
              rho(x) = 
              dx = 
              periodic = True/False 

          Returns:
              phi(x), type = vector with dimensions Nx x 1 

      ---------------------------------------------------------------------------
      """
    A = csc_matrix(diags([2, -1, -1], [0, -1, 1], shape = (rho.shape[0], rho.shape[0])).toarray())


    if periodic:
        A[-1, 0] = 1
        A[0, -1] = -1


    A /= (dx)**2

    # inversely solve the A @ phi == rho linear equation set for phi

    phi, exitcode = gmres(A, rho) # generalized minimum residual method for solving linear systems A phi = Rho
    print(exitcode) # if it prints out a zero, we've ensured successful convergence
    np.allclose(A.dot(phi), rho) # close the matrix, save memory

    return phi

def electric_field(phi, dx, periodic = True):
    """This function returns the spatially dependent electric field given phi(x)
    and dx = spatial step

    E =  d/dx (phi (x))
-------------------------------------------------------------------------------
    PARAMETERS:
        inputs:
            phi = electric potential, spatially dependent. Can be found by calling
            the electric_potential function. type = vec n x 1 
            
            dx = spatial step, type = int 
            
            periodic = True or False, indicating boundary conditions 
            
        returns:
            - E(x) = spatially dependent electric field, type: vector with dimensions Nx x 1 
            
            to obtain E(x), simply multiply by -1. 
-------------------------------------------------------------------------------
"""
    
            
    A = diags([0, -1, 1], [0, -1, 1], shape=(phi.shape[0], phi.shape[0])).toarray()
   
    if periodic:
       #A[0, 0] = -1
       #A[-2, -2] = 1
       #A[-1, -1] = -1
        
        A[0, -2] = -1
        #A[1, 0] = -1
        #A[0, 1]  = 1
        A[-1, 1] = 1
        #A[-2, -1] = 1
        #A[-1, -2] = -1
       
    else:
        A[0, 0] = -2
        A[0, 1] = 2
        A[-1,-1] = 2
        A[-1,-2] = 2 


    A /= (2*dx)

    return A @ phi

# Functions for the Hamiltonian
def Hamiltonian(V, phi):
    """This function returns the Hamiltonian in X, V summed for n particles
-------------------------------------------------------------------------------
    PARAMETERS:
        inputs:
            V = state vector containing n particles' velocities, n x 1 
            
            phi = spatially dependent electric potential, type = vector of n x 1 
            
            Returns:
                Hamiltonian = sum over i = 1 : n particles [ 1/2 V^2 + q/m phi(x)]
                
                q/m nominally taken to be = -1
-------------------------------------------------------------------------------
    
    """
    return 0.5 * (V @ np.ones(V.T.shape))**2 - (phi @ np.ones(phi.T.shape))

def Hamiltonian_x(phi, q_m_ratio):
    return q_m_ratio * (phi @ np.ones(phi.T.shape))

def Hamiltonian_v(V):
    return 0.5 * (V @ np.ones(V.T.shape))**2