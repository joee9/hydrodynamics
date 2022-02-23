# Joe Nyhan, 30 June
# Initial conditions for the hydrodynamical system.

from hd_params import *
from aux.hd_equations import R
from aux.hd_eos import P


def getvals(input):
    vals = []

    for i in range(NUM_VPOINTS):
        vals.append(0)

    with open(input, "r") as f:
        for i in range(NUM_SPOINTS-NUM_VPOINTS):
            val = f.readline()
            if (val != ""):    vals.append(val)
            else: vals.append(0)

    vals = np.array(vals)
    print(f'{input} successfully read.\n')

    return vals

# ========== PRIMITIVES

if PRIM_IC == "TOV solution polytrope":

    if charge:
        rho_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_dr{dr:.3f}_rho.txt"
        p_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_dr{dr:.3f}_P.txt"
    if not charge:
        rho_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_dr{dr:.3f}_rho.txt"
        p_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_dr{dr:.3f}_P.txt"

    initial_rho_vals = getvals(rho_file)
    initial_p_vals = getvals(p_file)

if PRIM_IC == "TOV solution fit":
    rho_file = f"input/{eos}_p{p_val:.8f}_dr{dr:.3f}_rho.txt"
    p_file = f"input/{eos}_p{p_val:.8f}_dr{dr:.3f}_P.txt"

    initial_rho_vals = getvals(rho_file)
    initial_p_vals = getvals(p_file)


@njit
def rho_gaussian(r):
    return A*np.exp(-(r-r0)**2/d**2)+1E-13

if PRIM_IC == "Gaussian":
    rho_vals = []
    p_vals = []

    for i in range(NUM_SPOINTS):
        r = R(i+1/2)    # stored at a staggered grid; i + 1/2 -> i
        rho_ = rho_gaussian(r)
        rho_vals.append(rho_)
        p_vals.append(P(rho_))
        
    initial_rho_vals = np.array(rho_vals)
    initial_p_vals = np.array(p_vals)

# ========== CONSERVATIVE FROM PRIMITIVE

@njit
def Pi(prim):
    """
    prim: the primitive values at a given spatial and temporal point
    """
    P, rho, v = prim
    return (rho + P) / (1 - v) - P

@njit
def Phi(prim):
    """
    prim: the primitive values at a given spatial and temporal point
    """
    P, rho, v = prim
    return (rho + P) / (1 + v) - P

# ========== GAUSSIAN
