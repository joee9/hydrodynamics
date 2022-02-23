# Joe Nyhan, 09 November 2021
# Run with: makevals.py in /static_solutions; will make corresponding
#   table to be used for interpolation during tov.py
from hd_params import *
from aux.hd_eos import P

# make file
rhos = np.logspace(-20,0,20000,base=10.0)

with open(f"./0-{eos}_vals.vals", "w") as f:
    for i in range(len(rhos)):
        f.write(f"{rhos[i]:.16e}, {P(rhos[i]):.16e}\n")

print("COMPLETED.")