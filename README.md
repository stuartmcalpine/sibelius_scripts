# SIBELIUS simulation read scripts

Python3 scripts to read data from SIBELIUS simulations.

Reads catalog information from HBT, Velociraptor and GALFORM outputs.

Install `python3 setup.py install` or `python3 setup.py install --user`

# link_sibelius

Each catalog object returns a data array of subhalo/galaxy quantities (see examples below). This data array can be appended with Sibelius-specific computed properties (such as RA/DEC etc..). To do this, call the link_sibelius() function (see examples below).

### Input params to the link_sibelius function (all params are optional)

| Input | Description |
| ----- | ----------- |
| compute_distance= | Compute distance from each galaxy to the MW |
| compute_ra_dec= | Compute RA DEC of each galaxy |
| compute_velocity= | Compute radial and tangential velocities of each galaxy relative to MW |
| compute_galactic= | Compute galactic coordinates to each galaxy |
| compute_apparent_mag= | Compute apparent magnitudes of each galaxy based on absolute mag and distance |
| observer= | Coordinate location of the observer. Default is the SIBELIUS-DARK MW subhalo. This can be overwritten with a 3d coordinate array. |

# read_galform

Simple python3 script to read GALFORM output for SIBELIUS simulations.

Output data in the GALFORM directory is expected to be multiple `ivol_XXX` folders, or `ivol_XXX_mags` folders which contain the magnitudes.

In the MPI reading case, each rank reads its own subset of the `ivol_XXX` file parts.

### Input params to read_galform

| Input | Description | Is optional? | Default option |
| ----- | ----------- | --------- | ------- | 
| data_dir | path to GALFORM data (containing `ivol_XXX` directories) | No | - |
| num_files | number of files GALFORM data is split over (number of `ivol_XXX` directories) | No | - |
| output_no | output/snapshot number | No | - | 
| comm= | MPI4PY communicator | Yes | None |
| verbose= | True for more stdout output | Yes | False |
| only_single_ivol= | Option to read single ivol file | Yes | None |
| to_convert_out_h= | True to convert out h-values | Yes | True |

### Example usage (No MPI case)

```python
import sibelius.galform as galform

# Set up read_galform object.
data_dir = "/path/to/galform/folder/"
num_files = 1024
output_no = 1 # z=0
g = galform.read_galform(data_dir, num_files, output_no)

# Load galaxy data.
what_to_load = ["SubhaloID", "xgal", "ygal", "zgal"]
g.load_galaxies(what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
g.link_sibelius(compute_distance=True)

# Access the data.
print(g.data["xgal"])
```

### Example usage (MPI case)

```python
from mpi4py import MPI
import sibelius.galform as galform

# MPI communicator.
comm = MPI.COMM_WORLD

# Set up read_galform object.
data_dir = "/path/to/galform/folder/"
num_files = 1024
output_no = 1 # z=0
g = galform.read_galform(data_dir, num_files, output_no, comm=comm)

# Load galaxy data.
what_to_load = ["SubhaloID", "xgal", "ygal", "zgal"]
g.load_galaxies(what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
g.link_sibelius(compute_distance=True)

# Reduce all galaxies to rank 0.
g.gather_galaxies()

# Access the data.
print(g.data["xgal"])
```

# read_hbt_subhaloes

Simple python3 script to read HBT+ output for SIBELIUS simulations.

`hbt_dir` is the parent directory, which contains subdirectories for each snapshot, in a format `XXX`, each of which contain the `SubSnap_XXX.xx.hdf5` files.

In the MPI reading case, each rank reads its own subset of the `SubSnap_XXX.xx.hdf5` files.

After loading the subhalo propeties, there is an additional option to compute SIBELIUS specific properties, such as RA, DEC, distance to the MW etc. 
These are all based upon the MW (and other objects) positions from the SIBELIUS-DARK production run. This will add additional information to the original data
dict (hbt.data in the examples below), see `sibelius_functions.py` for the output names within the dict (see table above for input options for this function).

### Input params to read_hbt_subhaloes.py

| Input | Description | Is optional? | Default option |
| ----- | ----------- | --------- | ------- | 
| hbt_dir | path to parent HBT+ data (containing `XXX` format snapshot directories) | No | - |
| comm= | MPI4PY communicator | Yes | None |
| verbose= | True for more stdout output | Yes | False |

### Example usage (No MPI case)

```python
from sibelius.hbt import read_hbt_subhaloes

# Set up HBT object.
hbt_dir = "/path/to/parent/hbt/folder/"
hbt = read_hbt_subhaloes(hbt_dir)

# Load subhalo data.
snapnum = 199 # z=0 for SIBELIUS simulations.
what_to_load = ['ComovingMostBoundPosition', 'Mbound', 'HostHaloId', 'Rank', 'Nbound']
hbt.load_haloes(snapnum, what_to_load=what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
hbt.link_sibelius(compute_distance=True)

# Access the data.
print(hbt.data["Mbound"])
```

### Example usage (MPI case)

```python
from mpi4py import MPI
from sibelius.hbt import read_hbt_subhaloes

# MPI communicator.
comm = MPI.COMM_WORLD

# Set up read_galform object.
hbt_dir = "/path/to/parent/hbt/folder/"
hbt = read_hbt_subhaloes(hbt_dir, comm=comm)

# Load subhalo data.
snapnum = 199 # z=0 for SIBELIUS simulations.
what_to_load = ['ComovingMostBoundPosition', 'Mbound', 'HostHaloId', 'Rank', 'Nbound']
hbt.load_haloes(snapnum, what_to_load=what_to_load)

# Link SIBELIUS specific properties (compute the distance to each object from the Milky Way).
hbt.link_sibelius(compute_distance=True)

# Reduce all galaxies to rank 0.
hbt.gather_haloes()

# Access the data.
print(hbt.data)
```

# read_subhalo_particles

HBT stores the particle IDs bound to each subhalo.

This example shows how to load the particle IDs for a given list of subhaloes
(identified by their TrackId's).

In MPI mode this function will still work, whatever TrackId's happen to be in
the file subparts that rank reads will be stored in that rank.

```python
from sibelius.hbt import read_hbt_subhaloes

# Set up HBT object.
hbt_dir = "/path/to/parent/hbt/folder/"
hbt = read_hbt_subhaloes(hbt_dir)

# Load subhalo data.
snapnum = 199 # z=0 for SIBELIUS simulations.
trackid_list = [1,2,3]
particle_ids_dict = hbt.load_subhalo_particles(snapnum, trackid_list)

# Access the data.
print(particle_ids_dict)
```

# read_hbt_subhalo_history

Simple python3 script to track subhaloes through time from the HBT+ output for SIBELIUS simulations.

Accepts a list of trackids and returns their HBT properties at each snapshot.

### Example usage (MPI case)

```python
from mpi4py import MPI
import sibelius.read_hbt_subhalo_history as subhalo_history

# MPI communicator.
comm = MPI.COMM_WORLD

# Set up read_galform object.
hbt_dir = "/path/to/parent/hbt/folder/"
hbt = subhalo_history(hbt_dir, comm=comm)

# Load subhalo histories.
snapnum = 199 # z=0 for SIBELIUS simulations (where to first find the trackids).
trackids = [1,2,3]
data = hbt.hbt.load_galaxy_history(trackids, snapnum)
```
