# Joe Nyhan, 30 June 2021
# The equations used for simulation.

# from logging import debug
from numba.core.decorators import jit
from hd_params import *
from aux.hd_eos import *
# NUMBA_DISABLE_JIT = 1

# ========== GRID

@njit
def R(i):
    """
    gives the grid point for a given index i; here, each R(i) is center of a cell, C_i, with domain [R((i+1)/2),R((i-1)/2)]
    """
    return (i - NUM_VPOINTS + 0) * dr

# ========== CALCULATE PRIMITIVES

@njit
def V(u, P):
    """
    from eq. (30); calculates v at a given i, given P and u
    """
    Pi, Phi = u
    # print(Pi,Phi,P)
    return (Pi - Phi)/(Pi + Phi + 2*P)

@njit
def calcPrims(u, rho0):
    """
    calculates the primitive values, given the conservative values at a given spatial point
    """
    rho_ = rho(u, rho0)
    p = P(rho_)
    v = V(u, p)

    return np.array([p,rho_,v])

# ========== TIME EVOLUTION

# fermiionic neutron star matter

@njit
def du_dt(i, rho0, u, a, a_p1h, a_m1h, alpha, alpha_p1h, alpha_m1h, F1_p1h, F1_m1h, F2_p1h, F2_m1h):
    """
    The time evolution of u
    F1, F2: the F value at a given spatial boundary (still a two component vector)
    """

    # calculate average alpha value for cell center
    alpha = (alpha_p1h + alpha_m1h)/2

    # r values
    r = R(i)
    r_p1h = R(i+1/2)
    r_m1h = R(i-1/2)

    firstFrac = 3/(r_p1h**3 - r_m1h**3)
    def fbTerm(r,alpha,a,F1):
        return r**2 * alpha * F1 / a

    firstBracket = (fbTerm(r_p1h, alpha_p1h, a_p1h, F1_p1h) - fbTerm(r_m1h, alpha_m1h, a_m1h, F1_m1h))
    
    def sbTerm(alpha, a, F2):
        return alpha * F2 / a

    secondBracket = (sbTerm(alpha_p1h,a_p1h,F2_p1h) - sbTerm(alpha_m1h,a_m1h,F2_m1h))

    s = S(u, r, alpha, a, rho0)

    return -firstFrac * firstBracket - 1/dr * secondBracket + s

@njit
def S(u,r,alpha,a,rho0):
    """
    the value of S at a grid
    alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
    """
    th = theta(u,r,alpha,a, rho0)
    return np.array([th, -th])

@njit
def theta(u, r, alpha, a, rho0):
    """
    the value of theta at a gridpoint
    alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
    """
    Pi, Phi = u
    rho_ = rho(u, rho0)
    p = P(rho_)
    v = V(u,p)
    firstBrackets = ((Pi-Phi)*v - (Pi+Phi))
    secondBrackets = (8 * np.pi * r * a * alpha * p + ((a * alpha)/(2*r))*(1-(1/a**2)))
    lastTerm = (a*alpha/(2*r)) * (1-(1/a**2))*p
    return (1/2) * firstBrackets * secondBrackets + lastTerm

# ========== GRAVITY FUNCTIONS

# initial a
@njit
def fa0(r, a, u):

    if r == 0:
        return 0
    else:
        Pi, Phi = u 
        return 4 * np.pi * r * a**3 * (Pi + Phi) / 2 - a * (a**2 - 1) / (2 * r)

# time evolution of a
@njit
def fa(i,alpha,a,cons):
    r = R(i)
    Pi, Phi = cons
    return -4 * np.pi * r * alpha * a**2 * (Pi-Phi)/2

# alpha (eq. 46)
@njit
def falpha(alpha_p1, r, cons, prim, a):
    Pi, Phi = cons
    P, rho, v = prim
    
    brackets = 1/2 * (Pi - Phi) * v + P
    braces = 4 * np.pi * r * a**2 * brackets + (a**2 - 1)/(2*r)
    exponential = np.exp(-dr*braces)
    return alpha_p1 * exponential
