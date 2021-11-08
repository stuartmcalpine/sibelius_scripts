# read_galform.py

Simple python3 script to read GALFORM output for SIBELIUS simulations.

Output data in the GALFORM directory is expected to be multiple `ivol_XXX` folders with additionally `ivol_XXX_mags` folders containing the magnitudes.

In the MPI reading case, each rank reads its own subset of the `ivol_XXX` files.

After loading the galaxy propeties, there is an additional option to compute SIBELIUS specific properties, such as RA, DEC, distance to the MW etc. 
These are all based upon the MW (and other objects) positions from the SIBELIUS-DARK production run. This will add additional information to the original data
dict (galform.data in the examples below), see `sibelius_functions.py` for the output names within the dict.

### Input params to read_galform.py

| Input | Description | Is optional? | Default option |
| ----- | ----------- | --------- | ------- | 
| data_dir | path to GALFORM data (containing `ivol_XXX` directories) | No | - |
| num_files | number of files GALFORM data is split over (number of `ivol_XXX` directories) | No | - |
| comm= | MPI4PY communicator | Yes | None |
| verbose= | True for more stdout output | Yes | False |

### Input params to the link_sibelius function (all params are optional and expect True or False)

| Input | Description |
| ----- | ----------- | 
| compute_distance= | Compute distance from each galaxy to the MW |
| compute_ra_dec= | Compute RA DEC of each galaxy |
| compute_velocity= | Compute radial and tangential velocities of each galaxy relative to MW |
| compute_galactic= | Compute galactic coordinates to each galaxy |
| compute_apparent_mag= | Compute apparent magnitudes of each galaxy based on absolute mag and distance |
| compute_extra_coordinates= | Simulation x and z axis are flipped. This returns `coords_eq` which is in true equitorial coordinates |
| compute_extra_objects= | Repeats all the computations above, but now also for M31, Coma and Virgo |

### Example usage (No MPI case)

```python
from read_galform import read_galform

# Set up read_galform object.
data_dir = "/path/to/galform/folder/"
num_files = 1024
galform = read_galform(data_dir, num_files)

# Load galaxy data.
what_to_load = ["SubhaloID", "xgal", "ygal", "zgal"]
galaxies = galform.load_galaxies(what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
galaxies.link_sibelius(compute_distance=True)
```

### Example usage (MPI case)

```python
from mpi4py import MPI
from read_galform import read_galform

# MPI communicator.
comm = MPI.COMM_WORLD

# Set up read_galform object.
data_dir = "/path/to/galform/folder/"
num_files = 1024
galform = read_galform(data_dir, num_files, comm=comm)

# Load galaxy data.
what_to_load = ["SubhaloID", "xgal", "ygal", "zgal"]
galaxies = galform.load_galaxies(what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
galaxies.link_sibelius(compute_distance=True)

# Reduce all galaxies to rank 0.
galform.gather_galaxies()
```
