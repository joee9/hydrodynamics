#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# static solutions
eos_UR = 0
eos_polytrope = 0
eos_SLy = 1

p0 = 1e-4
vphi0 = 1e-3
Lambda = 0
dr = 0.02
rmin = 0
rmax = 100

f_P     = 1
f_rho    = 0

if eos_polytrope:
    eos = "polytrope_"
    K = 100
    Gamma = 2
    params = f"K{K:.1f}_gamma{Gamma:.1f}_"
elif eos_SLy:
    eos = "SLy_"
    params = ""

s = f"../input/{eos}{params}p{p0:.8f}_dr{dr:.3f}_"

if f_P:
    s += "P.txt"
    df = pd.read_csv(s, header=None)
    title = "$P$"
elif f_rho:
    s += "rho.txt"
    df = pd.read_csv(s, header=None)
    title = "$\\rho$"

rs = np.arange(rmin, rmax, dr)
h_axis = rs
v_axis = df.iloc[:,0].to_numpy()[:len(rs)]
h_label = "$r$"

plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)

# %%

# Mass vs. P_0 or R
from scipy.interpolate import interp1d
from scipy.optimize import minimize, fmin
# analyze M vs. P(0) analyses from tov

M_vs_R = 1
M_vs_p0 = 0

path = "./p0_analysis"
# eos = "polytrope"
# eos = "SLy"
# eos = "FPS"
eos = "BSk19"
# eos = "BSk20"
# eos = "BSk21"

pmin = 1e-6
pmax = 1e-1

df = pd.read_csv(f"{path}/{eos},p{pmin:.3e}-p{pmax:.3e}.txt", header=None)

M_vals = df.iloc[:,1].to_numpy()
R_vals = df.iloc[:,2].to_numpy()  
p0_vals = df.iloc[:,0].to_numpy()

if M_vs_p0:
    tag = "p0"
    crit_p0_guess = p0_vals[np.argmax(M_vals)]

    p0_interp = interp1d(p0_vals,1/M_vals)
    p0_crit = fmin(p0_interp, crit_p0_guess)[0]
    M_crit = 1/(p0_interp(p0_crit))

    M_crit *= 1.63161 # km
    p0_crit /= 7.6804e-6 # MeV/fm^3

    plt.text(1e-3,.6,"$P_{crit} = $" + f"{p0_crit:.4e}")
    plt.text(1e-3,.5,"$M_{crit}$ = " + f"{M_crit:.4e}")
    plt.title(f"$M(P_0)$, {eos}")
    plt.xlabel("$P_0$")

    print(f"\nCritical pressure: {p0_crit:.4e}, Critical Mass: {M_crit:.4e}")

    h_axis = p0_vals / 7.6804e-6
    v_axis = M_vals * 1.63161
    plt.plot(p0_crit, M_crit, "ro")
    
if M_vs_R:
    tag = "r"
    crit_R_guess = R_vals[np.argmax(M_vals)]

    R_interp = interp1d(R_vals,1/M_vals)
    R_crit = fmin(R_interp, crit_R_guess)[0]
    M_crit = 1/(R_interp(R_crit))

    M_crit *= 1.63161 # km
    R_crit *= 2.4091 # km

    plt.text(8.5,.65,"$R_{crit} = $" + f"{R_crit:.4e}")
    plt.text(8.5,.5,"$M_{crit}$ = " + f"{M_crit:.4e}")
    plt.title(f"$M(R)$, {eos}")
    plt.xlabel("$R$")

    print(f"\nCritical radius: {R_crit:.4e}, Critical Mass: {M_crit:.4e}")

    h_axis = R_vals * 2.4091
    v_axis = M_vals * 1.63161
    plt.plot(R_crit, M_crit, "ro")
    plt.xlim(8,15)

# plt.xscale("log")
plt.plot(h_axis, v_axis)
plt.savefig(f"./plots/{tag}_analysis,{eos}.pdf", bbox_inches = "tight")

# %%

from scipy.interpolate import interp1d
# analyze M vs. P(0) analyses from tov for all EoSs

M_vs_R = 1
M_vs_p0 = 0

path = "./p0_analysis"
EoSs = [
    "SLy",
    "FPS",
    "BSk19",
    "BSk20",
    "BSk21"
]
styles = [
    "y-.", #Sly
    "g-",  #FPS
    "m--", #BSk19
    "r--", #BSk20
    "c-"   #BSk21
]

pmin = 1e-6
pmax = 1e-1

# for eos in EoSs:
for i in range(len(EoSs)):
    eos = EoSs[i]
    style = styles[i]
    # print(f"{eos}")
    df = pd.read_csv(f"{path}/{eos},p{pmin:.3e}-p{pmax:.3e}.vals", header=None)

    M_vals = df.iloc[:,1].to_numpy()
    R_vals = df.iloc[:,2].to_numpy()  
    p0_vals = df.iloc[:,0].to_numpy()

    if M_vs_p0:
        tag = "p0"
        plt.title(f"$M(P_0)$")
        plt.xlabel("$P_0$ [MeV/fm$^3$]")
        plt.xscale("log")
        plt.xlim(10**0, 10**4)

        h_axis = p0_vals / 7.6804e-6
        v_axis = M_vals * 1.63161
        
    if M_vs_R:
        tag = "r"
        crit_R_guess = R_vals[np.argmax(M_vals)]

        plt.title(f"$M(R)$")
        plt.xlabel("$R$, [km]")

        h_axis = R_vals * 2.4091
        v_axis = M_vals * 1.63161
        plt.xlim(5,20)

    plt.plot(h_axis, v_axis, f"{style}", label = f"{eos}")

# plt.legend(loc = "upper left")
plt.legend(loc = "upper right")
plt.savefig(f"./p0_analysis/{tag}_analysis,all.pdf", bbox_inches = "tight")