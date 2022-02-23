# Joe Nyhan, 30 June 2021; edited 20 July 2021, 20 September 2021, 15 November 2021
# For reading and plotting data fils from /data
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift

#%%

output_number = 1
save_fig = 0

# dim = "s"
dim = "ringdown"

# fft parameters
FFT = 1
FFT_spikes = 0     # reads in frequencies from the file path/freqs.txt for plotting
fft_xmin = .5e-1
fft_xmax = 1e0

# for "s", this is the time snapshot.
# for "ringdown", the max time plotted to. -1 will plot all values
t = -1

f_Pi    = 0
f_Phi   = 0

f_P     = 1
f_rho   = 0
f_v     = 0

f_alpha = 0
f_a     = 0


# ========== READING IN FILES, PARAMETERS

path = f"data/{output_number:d}"

with open(f"{path:s}-0params.txt", "r") as params:
    s = params.readline()
    interval = int(s.replace("Write interval    = ", ""))
    s = params.readline()
    ring_interval = int(s.replace("Ring interval     = ", ""))

dt = 0.01
i = round(t /(interval * dt))

# Plotting parameters
if f_Pi:
    file_path = f"{path:s}-Pi.txt"
    title = "$\\Pi$"
    save_name = "Pi"
    ring_idx = 2
if f_Phi:
    file_path = f"{path:s}-Phi.txt"
    title = "$\\Phi$"
    save_name = "Phi"
    ring_idx = 3
if f_P:
    file_path = f"{path:s}-P.txt"
    title = "$P$"
    save_name = "P"
    ring_idx = 4
if f_rho:
    file_path = f"{path:s}-rho.txt"
    title = "$\\rho$"
    save_name = "rho"
    ring_idx = 5
if f_v:
    file_path = f"{path:s}-v.txt"
    title = "$v$"
    save_name = "v"
    ring_idx = 6
if f_alpha:
    file_path = f"{path}-alpha.txt"
    title = "$\\alpha$"
    save_name = "alpha"
    ring_idx = 7
if f_a:
    file_path = f"{path:s}-a.txt"
    title = "$a$"
    save_name = "a"
    ring_idx = 8
    
save_name = f"plots/{output_number:d}-{save_name}"

if dim == "s":

    df = pd.read_csv(file_path, header=None)
    rs = pd.read_csv(f"{path:s}-r.txt", header=None)

    h_axis = rs.to_numpy().flatten()
    v_axis = df.iloc[i,:].to_numpy()
    h_label = "$r$"

    title += f", $t$ = {t:.1f}"
    save_name += f",{t:.2f}"

if dim == "ringdown":

    df = pd.read_csv(f"{path:s}-ringdown.txt", header=None)

    tmin = df.head(n=1).iloc[:,1].to_numpy()[0]
    tmax = df.tail(n=1).iloc[:,1].to_numpy()[0]

    if t == -1: 
        last_idx = round(tmax/(ring_interval * dt))
    else:
        last_idx = round(t/(ring_interval * dt))
        tmax = t

    title += f", $r = 0$"
    h_axis = df.iloc[:last_idx,1].to_numpy() # time
    v_axis = df.iloc[:last_idx,ring_idx].to_numpy() # pressure
    h_label = "t"

    save_name += f",ringdown"
    if t != -1: save_name += f",tmax{t}"

    if FFT:

        N = len(v_axis)
        FT = fftshift(fft(v_axis))

        dt = df.head(n=3).iloc[:,1].to_numpy()[2] - df.head(n=2).iloc[:,1].to_numpy()[1]

        Df = 1/(tmax-tmin)
        freqs = np.arange(-1/(2*dt), +1/(2*dt), Df) * 2*np.pi
        plt.xscale("log")
        plt.yscale("log")

        h_axis = freqs
        v_axis = np.abs(FT)

        save_name += f",fft"
        if t != -1:
            title += ", $t_{max}$ = " + f"{t}" 
        title += ", FFT"
        h_label = f"$f$"

        if FFT_spikes:
            vals = []
            with open(f"{path:s}-freqs.txt") as f:
                val = f.readline()
                while (val != ""):
                    val = float(val)
                    vals.append(val)
                    val = f.readline()
                
            vals = np.array(vals)
            for i in range(len(vals)):
                plt.plot([vals[i], vals[i]], [1e-6,1e-1])
        
        plt.xlim(fft_xmin, fft_xmax)



title += f", Trial: {output_number:d}"
save_name += f".pdf"


plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)
        
if save_fig:
    plt.savefig(save_name, bbox_inches = "tight")
