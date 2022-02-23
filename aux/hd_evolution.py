# Joe Nyhan, 08 July 2021
# Evolution equations for dark matter scalar field.

from hd_params import *
from aux.hd_equations import *
from aux.hd_riemann import *
from aux.hd_ops import *

@njit
def calc_gravity_functions(a, alpha, u, prim):
    """
    make sure to pass all arrays at the current timestep, or with corresponding kvalues already included
    """

    h = dr
    a[NUM_VPOINTS] = 1

    # calculate a throughout using modified Euler
    for i in range(NUM_VPOINTS, NUM_SPOINTS-1):
        k1 = h * fa0(R(i), a[i], u[i,:])
        k2 = h * fa0(R(i)+h, a[i] + k1, u[i+1,:])

        a[i+1] = a[i] + (1/2) * (k1 + k2)
    
    # calculate alpha, moving from the back towards r = 0

    alpha[NUM_SPOINTS-1] = (1/2 * (a[NUM_SPOINTS-1] + a[NUM_SPOINTS-2]))**(-1) # average of last two a values
    for i in range(NUM_SPOINTS-1, NUM_VPOINTS, -1): # start at last boundary and move inwards
        alpha[i-1] = falpha(alpha[i], R(i), u[i,:], prim[i,:], a[i])

    initializeEvenVPs(a)
    initializeEvenVPs(alpha, spacing="staggered")

# calculate k values in evolution
@njit
def calculate_kval(h, ku, alpha, a, u, F1, F2, rhos):

    for i in range(NUM_VPOINTS, NUM_SPOINTS-2):
        # print(f"{i=}\n")

        alpha_m1h = alpha[i-1]
        alpha_p1h = alpha[i]
        # alpha_p3h = alpha[i+1]
        alpha_ = (alpha_p1h + alpha_m1h)/2

        a_ = a[i]
        a_p1h = (a[i] + a[i+1])/2
        a_m1h = (a[i] + a[i-1])/2
        if i>NUM_VPOINTS:
            ku[i,:] = h * du_dt(i, rhos[i], u[i,:], a_, a_p1h, a_m1h, alpha_, alpha_p1h, alpha_m1h, F1[i,:], F1[i-1,:], F2[i,:], F2[i-1,:])


@njit
def evolution_rk3(cons,prim,a,alpha,curr,next):

    # Pi, Phi across space (i, func) at a given timestep
    u = cons[curr,:,:]

    rhos = prim[curr,:,1] # used for Newton-Rhapson method as guesses

    # ========== RK3 ========== #
    h = dt

    # cell reconstruction for the grid
    # compute uL and uR for a given u; this is done for all space (shape: NUM_SPOINTS,2)
    uL = np.zeros((NUM_SPOINTS,2))        # u as calculated at the cell boarder using the left values
    uR = np.zeros((NUM_SPOINTS,2))         # u as calculated at C.B. using the right values
    cell_reconstruction(u, uL, uR)

    # print(uR[NUM_SPOINTS-10:,:])
    F1 = np.zeros((NUM_SPOINTS,2))
    F2 = np.zeros((NUM_SPOINTS,2))

    # calculate all fluxes
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
        F1[i,:], F2[i,:] = findFluxes(i, uL[i,:], uR[i,:], rhos[i], rhos[i-1], rhos[i+1])

    
    # initialize all k values
    k1u = np.zeros((NUM_SPOINTS,2))
    k2u = np.zeros((NUM_SPOINTS,2))
    k3u = np.zeros((NUM_SPOINTS,2))

    # calculate k1
    calculate_kval(h, k1u, alpha[curr,:], a[curr,:], u, F1, F2, rhos)

    # intermediate calculations
    uk1 = u+k1u
    # floor the updated u values
    checkFloor(uk1[:,Pi_i])
    checkFloor(uk1[:,Phi_i])

    # inner boundary for updated u values
    Phi2 = uk1[2,Pi_i]
    Phi3 = uk1[3,Phi_i]
    r2 = R(2)
    r3 = R(3)
    uk1[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    uk1[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    initializeEvenVPs(uk1)

    # outer boundary for updated u values
    uk1[NUM_SPOINTS-1] = uk1[NUM_SPOINTS-3]
    uk1[NUM_SPOINTS-2] = uk1[NUM_SPOINTS-3]

    # cell reconstruction
    uLk1 = np.zeros((NUM_SPOINTS,2))        # u as calculated at the cell boarder using the left values
    uRk1 = np.zeros((NUM_SPOINTS,2))         # u as calculated at C.B. using the right values
    cell_reconstruction(uk1,uLk1,uRk1)

    F1k1 = np.zeros((NUM_SPOINTS,2))
    F2k1 = np.zeros((NUM_SPOINTS,2))

    # calculate all fluxes
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
        F1k1[i,:], F2k1[i,:] = findFluxes(i, uLk1[i,:], uRk1[i,:], rhos[i], rhos[i-1], rhos[i+1])

    # calculate k2
    calculate_kval(h, k2u, alpha[curr,:], a[curr,:], uk1, F1k1, F2k1, rhos)

    uk12 = u+(k1u + k2u)/4
    # floor the updated u values
    checkFloor(uk12[:,Pi_i])
    checkFloor(uk12[:,Phi_i])

    # inner boundary for updated u values
    Phi2 = uk12[2,Pi_i]
    Phi3 = uk12[3,Phi_i]
    r2 = R(2)
    r3 = R(3)
    uk12[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    uk12[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    initializeEvenVPs(uk12)

    # outer boundary for updated u values
    uk12[NUM_SPOINTS-1] = uk12[NUM_SPOINTS-3]
    uk12[NUM_SPOINTS-2] = uk12[NUM_SPOINTS-3]

    uLk12 = np.zeros((NUM_SPOINTS,2))        # u as calculated at the cell boarder using the left values
    uRk12 = np.zeros((NUM_SPOINTS,2))         # u as calculated at C.B. using the right values
    cell_reconstruction(uk12,uLk12,uRk12)

    F1k12 = np.zeros((NUM_SPOINTS,2))
    F2k12 = np.zeros((NUM_SPOINTS,2))

    # calculate all fluxes
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
        F1k12[i,:], F2k12[i,:] = findFluxes(i, uLk12[i,:], uRk12[i,:], rhos[i], rhos[i-1], rhos[i+1])
        

    #calc k3
    calculate_kval(h, k3u, alpha[curr,:], a[curr,:], uk12, F1k12, F2k12, rhos)

    # conservative functions
    for i in range(NUM_VPOINTS+1,NUM_SPOINTS-2):
        cons[next,i,:] = cons[curr,i,:] + 1/6 * (k1u[i,:] + k2u[i,:] + 4*k3u[i,:])
        

    checkFloor(cons[next,:,Pi_i])
    checkFloor(cons[next,:,Phi_i])

    cons[next, NUM_SPOINTS-1,:] = cons[next, NUM_SPOINTS-3,:]
    cons[next, NUM_SPOINTS-2,:] = cons[next, NUM_SPOINTS-3,:]

    u = cons[next,:,:]

    Phi2 = cons[next,2,Phi_i]
    Phi3 = cons[next,3,Phi_i]
    r2 = R(2)
    r3 = R(3)
    cons[next,1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    cons[next,1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)

    # virtual points
    initializeEvenVPs(cons[next,:,Pi_i])
    initializeEvenVPs(cons[next,:,Phi_i])
    

    # ========== CALCULATE PRIMITIVES ========== #

    for i in range(NUM_VPOINTS, NUM_SPOINTS):
        u = cons[next,i,:]
        rho_ = rho(u, rhos[i])
        p = P(rho_)
        v = V(u,p)
        prim[next,i,P_i] = p
        prim[next,i,rho_i] = rho_
        prim[next,i,v_i] = v 

    initializeEvenVPs(prim[next,:,P_i])
    initializeEvenVPs(prim[next,:,rho_i])

    # ========== CALCULATE GRAVITY FUNCTIONS ========== #

    calc_gravity_functions(a[next,:], alpha[next,:], cons[next,:,:], prim[next,:,:])

@njit
def evolution_modified_euler(cons,prim,a,alpha,curr,next):

    # Pi, Phi across space (i, func) at a given timestep
    u = cons[curr,:,:]

    rhos = prim[curr,:,1] # used for Newton-Rhapson method as guesses

    # ========== RK3 ========== #
    h = dt

    # cell reconstruction for the grid
    # compute uL and uR for a given u; this is done for all space (shape: NUM_SPOINTS,2)
    uL = np.zeros((NUM_SPOINTS,2))        # u as calculated at the cell boarder using the left values
    uR = np.zeros((NUM_SPOINTS,2))         # u as calculated at C.B. using the right values
    cell_reconstruction(u, uL, uR)

    # print(uR[NUM_SPOINTS-10:,:])
    F1 = np.zeros((NUM_SPOINTS,2))
    F2 = np.zeros((NUM_SPOINTS,2))

    # calculate all fluxes
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
        F1[i,:], F2[i,:] = findFluxes(i, uL[i,:], uR[i,:], rhos[i], rhos[i-1], rhos[i+1])

    
    # initialize all k values
    k1u = np.zeros((NUM_SPOINTS,2))
    k2u = np.zeros((NUM_SPOINTS,2))
    k3u = np.zeros((NUM_SPOINTS,2))

    # calculate k1
    calculate_kval(h, k1u, alpha[curr,:], a[curr,:], u, F1, F2, rhos)

    # intermediate calculations
    uk1 = u+k1u
    # floor the updated u values
    checkFloor(uk1[:,Pi_i])
    checkFloor(uk1[:,Phi_i])

    # inner boundary for updated u values
    Phi2 = uk1[2,Pi_i]
    Phi3 = uk1[3,Phi_i]
    r2 = R(2)
    r3 = R(3)
    uk1[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    uk1[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    initializeEvenVPs(uk1)

    # outer boundary for updated u values
    uk1[NUM_SPOINTS-1] = uk1[NUM_SPOINTS-3]
    uk1[NUM_SPOINTS-2] = uk1[NUM_SPOINTS-3]


    # cell reconstruction
    uLk1 = np.zeros((NUM_SPOINTS,2))        # u as calculated at the cell boarder using the left values
    uRk1 = np.zeros((NUM_SPOINTS,2))         # u as calculated at C.B. using the right values
    cell_reconstruction(uk1,uLk1,uRk1)

    F1k1 = np.zeros((NUM_SPOINTS,2))
    F2k1 = np.zeros((NUM_SPOINTS,2))

    # calculate all fluxes
    for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
        F1k1[i,:], F2k1[i,:] = findFluxes(i, uLk1[i,:], uRk1[i,:], rhos[i], rhos[i-1], rhos[i+1])

    # calculate k2
    calculate_kval(h, k2u, alpha[curr,:], a[curr,:], uk1, F1k1, F2k1, rhos)


    # conservative functions
    for i in range(NUM_VPOINTS+1,NUM_SPOINTS-2):
        cons[next,i,:] = cons[curr,i,:] + 1/2 * (k1u[i,:] + k2u[i,:])
        

    checkFloor(cons[next,:,Pi_i])
    checkFloor(cons[next,:,Phi_i])

    cons[next, NUM_SPOINTS-1,:] = cons[next, NUM_SPOINTS-3,:]
    cons[next, NUM_SPOINTS-2,:] = cons[next, NUM_SPOINTS-3,:]

    u = cons[next,:,:]

    Phi2 = cons[next,2,Phi_i]
    Phi3 = cons[next,3,Phi_i]
    r2 = R(2)
    r3 = R(3)
    cons[next,1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
    cons[next,1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)

    # virtual points
    initializeEvenVPs(cons[next,:,Pi_i])
    initializeEvenVPs(cons[next,:,Phi_i])
    

    # ========== CALCULATE PRIMITIVES ========== #

    for i in range(NUM_VPOINTS, NUM_SPOINTS):
        u = cons[next,i,:]
        rho_ = rho(u, rhos[i])
        p = P(rho_)
        v = V(u,p)
        prim[next,i,P_i] = p
        prim[next,i,rho_i] = rho_
        prim[next,i,v_i] = v 

    initializeEvenVPs(prim[next,:,P_i])
    initializeEvenVPs(prim[next,:,rho_i])

    # ========== CALCULATE GRAVITY FUNCTIONS ========== #

    calc_gravity_functions(a[next,:], alpha[next,:], cons[next,:,:], prim[next,:,:])
