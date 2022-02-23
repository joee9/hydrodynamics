# Joe Nyhan, 30 June 2021
# Functions for solving the Riemann problem using HLLE.

from hd_params import *
from aux.hd_ops import initializeEvenVPs
from aux.hd_equations import R, calcPrims
from aux.hd_eos import dP_drho

# ========== SLOPE LIMITER

@njit
def minmod(a,b):
    if a*b <= 0: return 0
    elif np.abs(a) > np.abs(b): return b
    else: return a


# ========== CELL RECONSTRUCITON

@njit
def cell_reconstruction(u, uL, uR):

    s = np.zeros((NUM_SPOINTS,2))        # the arguments used to calcluate the "minmod slope limiter"
    sigma = np.zeros((NUM_SPOINTS,2))    # the sigma values used for cell reconstruction

    rs = np.zeros(NUM_SPOINTS)

    for i in range(NUM_SPOINTS):
        rs[i] = R(i)
    # for the next equations, we always only index the arrays from NUM_VPOINTS to
    # NUM_SPOINTS - 2, as the last two indices are not used (and the first index too), as 
    # our convention and lack of information make force it to be that way

    # defined in eq. (64); a vector
    #
    # here, because values are stored at the gridpoint (index) to the left of where they
    #     are calculated, s_i stores the value at s_{i+1/2}
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2):
        # s[i,:] = (u[i+1,:] - u[i,:]) / (R(i+1) - R(i))
        s[i,:] = (u[i+1,:] - u[i,:]) / (rs[i+1] - rs[i])
    
    # here we calculate the sigma values for a given function and i value
    #
    # this loop runs through both functions for each i value and calculates the sigma
    #     for that given function
    #
    # as stated before, s_i and s_{i-1} are really s_{i+1/2} and s_{i-1/2}, but they are
    #     really stored at the index to the left, hence i and i-1
    #
    # see eq. (65) for definition
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2):
        for k in [0,1]: # (Pi, Phi)
            sigma[i,k] = minmod(s[i,k], s[i-1,k])
    
    # # as defined in eq. (68)
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2):
        r = rs[i]
        r_p1 = rs[i+1]
        r_p1h = (rs[i+1] + rs[i])/2
        uL[i,:] = u[i] + sigma[i,:] * (r_p1h - r)
        uR[i,:] = u[i+1] + sigma[i+1,:] * (r_p1h - r_p1)


    initializeEvenVPs(uL[:,0], "staggered")
    initializeEvenVPs(uL[:,1], "staggered")
    initializeEvenVPs(uR[:,0], "staggered")
    initializeEvenVPs(uR[:,1], "staggered")

# ========== FLUXES
    
@njit
def findFluxes(i, uL, uR, rho0, rho0_left, rho0_right):

    # rho0 values at the two boundaries
    rho0_p1h = 1/2 * (rho0 + rho0_right)
    rho0_m1h = 1/2 * (rho0 + rho0_left)

    prims_uL = calcPrims(uL, rho0_m1h)
    prims_uR = calcPrims(uR, rho0_p1h)

    lambda1_L = lambda1(uL, prims_uL)
    lambda2_L = lambda2(uL, prims_uL)
    lambda1_R = lambda1(uR, prims_uR)
    lambda2_R = lambda2(uR, prims_uR)

    vals = [lambda1_L, lambda2_L,lambda1_R, lambda2_R,0]

    lp = max(vals)        # lambda+, the max of the four calculated eigenvals and 0
    lm = min(vals)        # lambda-

    f1L = f1(uL, prims_uL)
    f2L = f2(uL, prims_uL)
    f1R = f1(uR, prims_uR)
    f2R = f2(uR, prims_uR)

    F1 = (lp * f1L - lm * f1R + lp * lm * (uR - uL)) / (lp - lm)
    F2 = (lp * f2L - lm * f2R) / (lp - lm)

    arr = F1, F2

    return arr


@njit
def drho_dP(rho,p):
    return 1/dP_drho(rho,p)

@njit
def dP_dPi(u,prims):
    Pi, Phi = u
    p, rho, v = prims
    num = rho - p - 2*Phi
    den = (Pi + Phi - 2*rho) - drho_dP(rho,p) * (Pi + Phi + 2*p)
    return num/den

@njit
def dP_dPhi(u,prims):
    Pi, Phi = u
    p, rho, v = prims
    num = rho - p - 2*Pi
    den = (Pi + Phi - 2*rho) - drho_dP(rho,p) * (Pi + Phi + 2*p)
    return num/den

@njit
def A11(u, prims):
    v = prims[2]
    return (1/2) * (1 + 2*v - v**2) + (1-v**2) * dP_dPi(u,prims)

@njit
def A12(u, prims):
    v = prims[2]
    return -(1/2) * (1+v)**2 + (1-v**2) * dP_dPhi(u,prims)

@njit
def A21(u, prims):
    v = prims[2]
    return (1/2) * (1-v)**2 - (1-v**2) * dP_dPi(u,prims)

@njit
def A22(u, prims):
    v = prims[2]
    return (1/2) * (-1 + 2*v + v**2) - (1-v**2) * dP_dPhi(u,prims)

@njit
def makeMatrix(u,prims):
    return np.array([[A11(u,prims), A12(u,prims)],[A21(u,prims), A22(u,prims)]])

@njit
def tr(A):
    a11 = A[0,0]
    a22 = A[1,1]
    return a11 + a22

@njit
def det(A):
    a11 = A[0,0]
    a12 = A[1,0]
    a21 = A[0,1]
    a22 = A[1,1]
    return a11 * a22 - a12 * a21

@njit
def lambda1(u, prims):
    A = makeMatrix(u,prims)
    return 1/2 * (tr(A) + np.sqrt((tr(A))**2 -4*det(A)))

@njit
def lambda2(u, prims):
    A = makeMatrix(u,prims)
    return 1/2 * (tr(A) - np.sqrt((tr(A))**2 -4*det(A)))

# fluxes

@njit
def f1(u, prims):
    Pi, Phi = u
    p, rho, v = prims

    comp1 = (1/2) * (Pi - Phi) * (1 + v)
    comp2 = (1/2) * (Pi - Phi) * (1 - v)

    return np.array([comp1, comp2])

@njit
def f2(u, prims):
    Pi, Phi = u
    p, rho, v = prims
    return np.array([p,-p])
