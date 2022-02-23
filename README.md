## Hydrodynamical neutron star simulation

This project is the simulation of a neutron star, with the star modelled as a fluid.

### Code
Broken into "auxillary" files (like `C` headers) and the main simulation file. Global parameters and variables are declared in `hd_params.py`.

### Flags/Options
All flags and parameters are consolidated to the `hd_params.py` file in root.

#### Recording
Data recording can be toggled with the `record_data` flag. Currently, this record data flag is configured to write data to a file for every parameter. The `output_number` flag will add any integer to the start of the file names for an individual run to allow for multiple runs to be stored and distinguished from one another. The `record_ringdown` flag creates an output file that stores all of the field values at one spatial point at more temporal points. Currently, values are stored at r = 0 (or, for staggered fields, r = 1/2 \* dr.


### I/O
Make sure that the folders `input` and `data` are created such that the program can write correctly. Otherwise, and error will occur.
*To be included in the future*: the static solutions will automatically be generated to the correct directory to be read by the main simulation code. Depending on the chosen initial conditions, the correct static solutions will be chosen.

## Static Solutions
All of the code for creating "static solutions" to be loaded into the temporal simulation is contained in the `./static_solutions` directory. Here, there are two main files of interest: `tov.py` and `makevals.py`.

### makevals.py
Use this file to tabulate values of $\rho$ and $P$ for use in `tov.py`. Set the EoS of choice within `hd_params.py` and run `python makevals.py` from within `./static_solutions`. A corresponding `.vals` file will be produced and placed within that directory to be read and dealt with by `tov.py`.

### tov.py
This file is run to create a static solution for a given equation of state. In order to use a given equation of state, however, we must have a way to get energy density, $\rho$, from $P$, pressure. This is the opposite of the way that it we implemented the EoSs in `aux/hd_eos.py`, so we instead tabulate values to be used within tov.py. This is what `makevals.py` is for.
