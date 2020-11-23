# neuralThermalAnisotropicIsing

This is a collection of programs intended for the purpose of using unsupervised learning to detect structural changes in an Ising model with anisotropic temperatures as the temperatures are varied.

This includes a program for Monte Carlo simulation of the system as well as programs for building and training a β-TCVAE as well as an InfoGAN.

Note that the neural networks are intended for use with TensorFlow-DirectML, but they should be compatible with Tensorflow 1.x and (with minor modification) Tensorflow 2.x.

## `monte_carlo.py`

### Requirements

- NumPy

- Numba

- tqdm

### Optional requirements for parallelization

- Dask

- Joblib

### Purpose

This is used for running a Monte Carlo simulation of an Ising model with different temperatures along the vertical and horizontal interactions, which introduces anistropy to the system.

### Usage

- `v`: Verbose mode

- `r`: Restart mode

- `p`: Parallel mode

- `c`: Client mode (for parallelization with Dask)

- `d`: Distributed mode (for PBS cluster with Dask)

- `rd`: Retart dump frequency (number of steps between restart file dumps)

- `rn`: Restart name (name prefix of restart file)

- `rs`: Retart step (step to restart from)

- `q`: Job queue (for distributed mode with Dask)

- `a`: Allocation (for distributed mode with Dask)

- `nn`: Nodes (for distributed mode with Dask)

- `np`: Processors per node (for distributed mode with Dask)

- `w`: Walltime (for distributed mode with Dask)

- `m`: Memory (for distributed mode with Dask)

- `nw`: Workers (for multiprocessing)

- `nt`: Threads (threads per worker)

- `mt`: Parallelization method (for parallelization with Dask)

- `nm`: Name (name prefix of simulation)

- `n`: Lattice sites (linear lattice size)

- `j`: Interaction (coefficient for nearest neighbor interactions)

- `txn`: T_x number (number of temperatures for horizontal system)

- `txr`: T_x range (minimum and maximum T_x)

- `tyn`: T_y number (number of temperatures for vertical system)

- `tyr`: T_y range (minimum and maximum T_y)

- `sc`: Sample cutoff (samples are only recorded after sample cutoff)

- `sn`: Sample number (total number of samples simulated including cutoff)

- `rec`: REMCMC cutoff (REMCMC or parallel tempering only performed before cutoff)

### Examples

Here are some examples for simulating a system with linear size l=81 with J=1.0 and 33 values each for T_x and T_y from 0.02 to 8.02. Units are the reduced Ising units. A total of 2048 samples for each temperature combination are generated, though only the last 1024 are recorded, with the first 1024 subject to replica exchange Markov Chain Monte Carlo (REMCMC) moves to aid with equilibration. For each step, all spins are subject to a flipping attempt with the Metropolis-Hastings criterion (exponentiated negative energy difference) in a random order. At the end of each move, the congifuration is subnject to a total flip with probability 0.5 due to the Z_2 symmetry of the system. After all samples are generated for a step, they are all subject to REMCMC moves (or parallel tempering). This is done by attempting to swap two configurations using the Metropolis Hastings Criterion (exponentiated enthalpy difference).

Serial:

`python monte_carlo.py -v -rd 128 -nm test_0 -n 81 -j 1.0 -txn 33 -txr 0.02 8.02 -tyn 33 -tyr 0.02 8.02 -sc 1024 -sn 2048 -rec 1024`

Multiprocessing with Dask:

`python monte_carlo.py -v -c -rd 128 -nw 24 -nt 1 -mt forkserver -nm test_0 -n 81 -j 1.0 -txn 33 -txr 0.02 8.02 -tyn 33 -tyr 0.02 8.02 -sc 1024 -sn 2048 -rec 1024`

Multithreading with Joblib:

`python monte_carlo.py -v -p -rd 128 -nt 24 -nm test_0 -n 81 -j 1.0 -txn 33 -txr 0.02 8.02 -tyn 33 -tyr 0.02 8.02 -sc 1024 -sn 2048 -rec 1024`

Here is an example for restarting from the last equilibration step (1024) of test_0 to generate and record 1024 samples.

Restarting simulation with Dask:

`python monte_carlo.py -v -c -r -rd 128 -rn test_0 -rs 1024 nw 24 -nt 1 -mt forkserver -nm test_1 -n 81 -j 1.0 -txn 33 -txr 0.02 8.02 -tyn 33 -tyr 0.02 8.02 -sc 0 -sn 1024 -rec 0`

## `parse_monte_carlo_output.py`

### Purpose

This program parses and saves NumPy arrays of the output of a Monte Carlo simulation.

### Requirements

- NumPy

### Usage

- `v`: Verbose mode

- `nm`: Name (name prefix of simulation)

- `n`: Lattice sites (linear lattice size)

### Example

Parsing the output of `test_0`:

`python parse_monte_carlo_output.py -v -nm test_0 -n 81`

## `plot_monte_carlo_thermal.py`

### Purpose

This program plots the various thermal properties of the Monte Carlo simulation output, including energies, magnetizations, specific heat capacities, and magnetic susceptibilities.

### Requirements

- NumPy

- Matplotlib

### Usage

- `v`: Verbose mode

- `nm`: Name (name prefix of simulation)

- `n`: Lattice sites (linear lattice size)

### Example

Plotting the output of `test_0`:

`python plot_monte_carlo_thermal.py -v -nm test_0 -n 81`