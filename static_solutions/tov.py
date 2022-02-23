# Joe Nyhan, 15 July 2021
# File for computing static solutions for fermionic stars via the TOV model.
#%%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
# from scipy.optimize import minimize, fmin
import matplotlib.pyplot as plt
# %%

# EOS
eos_UR        = 0
eos_polytrope = 0

eos_SLy       = 1
eos_FPS       = 0

eos_BSk19     = 0
eos_BSk20     = 0
eos_BSk21     = 0

if eos_UR:
    eos = "UR"
    Gamma = 1.3
if eos_polytrope:
    eos = "polytrope"
    Gamma = 2
    K = 100
if eos_SLy:
    eos = "SLy"
if eos_FPS:
    eos = "FPS"
if eos_BSk19:
    eos = "BSk19"
if eos_BSk20:
    eos = "BSk20"
if eos_BSk21:
    eos = "BSk21"

# MODES
make_static_solution = 0
p0_analysis = 1

# PARAMETERS

if make_static_solution:
    p0 = 1e-4

if p0_analysis:
    pmin = 1e-6
    pmax = 1e-1
    NUM_POINTS = 1000
    # p0_vals = np.arange(1e-6, 1e-1,5e-5)
    p0_vals = np.logspace(round(np.log10(pmin)),round(np.log10(pmax)),NUM_POINTS, base=10.0)

r0 = 0.000001
rmax = 200
dr = 0.02


p_tol = 1e-11
r_vals = np.arange(r0,rmax,dr)

m0 = 0

if make_static_solution: path = "../input"
elif p0_analysis: path = "./p0_analysis"

# FUNCTIONS

vals_path = ""
if eos_UR or eos_polytrope:
    pass
else:
    vals_path = f"0-{eos}_vals.vals"

    df = pd.read_csv(vals_path)
    interp_rhos = df.iloc[:,0].to_numpy()
    interp_ps = df.iloc[:,1].to_numpy()


rho_from_P_interp = interp1d(interp_ps, interp_rhos, fill_value="extrapolate")

def rho_from_P(p):
    if eos_UR:
        return p/(Gamma-1)
    elif eos_polytrope:
        return (p/K)**(1/Gamma)
    else:
        return rho_from_P_interp(p)

# def rho_from_P_UR(p):
#     return p/(Gamma-1)

# def rho_from_P_poly(p):
#     return (p/K)**(1/Gamma)

# rho_from_P_SLy = interp1d(SLy_ps, SLy_rhos)
        

def event(r,y):
    m, p = y

    return p - p_tol

event.terminal = True
event.direction = 0

def f_onefluid(r,y):
    m, p = y

    # if eos_UR:
    #     rho =  rho_from_P_UR(p)
    # elif eos_polytrope:
    #     rho = rho_from_P_poly(p)
    # elif eos_SLy:
    #     rho = rho_from_P_SLy(p)

    rho = rho_from_P(p)

    N = 1-2*m/r

    fm = 4*np.pi*r**2*rho
    frho = (-(4*np.pi*r**3*p + m)*(rho+p)/(r**2*N))

    return fm, frho

# ONE STATIC SOLUTIION

if make_static_solution:
    if eos_polytrope:
        P_path = f"{path}/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p0:.8f}_dr{dr:.3f}_P.txt"
        rho_path = f"{path}/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p0:.8f}_dr{dr:.3f}_rho.txt"

    # if eos_SLy:
    else:
        P_path = f"{path}/{eos}_p{p0:.8f}_dr{dr:.3f}_P.txt"
        rho_path = f"{path}/{eos}_p{p0:.8f}_dr{dr:.3f}_rho.txt"

    P_out = open(P_path,"w")
    rho_out = open(rho_path,"w")

    sol = solve_ivp(f_onefluid,[r0,rmax], [m0,p0], method='RK45', atol = 1e-12, rtol = 1e-12, t_eval=r_vals, events=event)

    m, p = sol.y

    for i in range(len(r_vals)):
        if i < len(p):
            P_out.write(f"{p[i]:.16e}\n")
            rho_out.write(f"{rho_from_P(p[i]):.16e}\n")
        else:
            P_out.write(f"{0:.16e}\n")
            rho_out.write(f"{0:.16e}\n")
    
    P_out.close()
    rho_out.close()

    print(f"COMPLETED.")
        
    

# P0 ANALYSIS

if p0_analysis:
    output = open(f"{path}/{eos},p{pmin:.3e}-p{pmax:.3e}.vals","w")

    print(f"Number of points:{len(p0_vals)}; for EoS: {eos}")

    def evolution(p0_vals):
        for i in range(len(p0_vals)):
            
            p0 = p0_vals[i]
            sol = solve_ivp(f_onefluid,[r0,rmax], [m0,p0], method='RK45', atol = 1e-12, rtol = 1e-12, t_eval=r_vals, events=event)
            M = sol.y[0][-1]
            R = sol.t[-1]

            if M < 0: print(f"M <0"); break

            output.write(f"{p0:.6e},{M:.6e},{R:.6e}\n")

            if i % 10 == 0: print(f"Timestep {i:3d} reached, {M=}.")

    evolution(p0_vals)

    output.close()