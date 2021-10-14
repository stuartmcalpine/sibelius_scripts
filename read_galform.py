import numpy as np
import h5py
import os 

class read_galform:

    def __init__(self, data_dir, num_files, comm=None, verbose=True):

        self.data_dir   = data_dir
        self.num_files  = num_files
        self.verbose    = verbose

        # MPI stuff.
        if comm is None:
            self.comm_rank = 0
            self.comm_size = 1
        else:
            self.comm = comm
            self.comm_rank = comm.rank
            self.comm_size = comm.size

    def load_galaxies(self, what_to_load):

        assert type(what_to_load) == list, "what_to_load not a list"
        self.data = {}

        # Loop over each file part.
        for i in range(self.num_files):
            if i % self.comm_size != self.comm_rank: continue

            # Open this file part (galaxies).
            this_f = f"{self.data_dir}/ivol_{i}/galaxies.hdf5"
            assert os.path.isfile(this_f), f"File {this_f} doesn't exist"
            if self.verbose: print(f"[Rank {self.comm_rank}] Loading file {i}...")
            f = h5py.File(this_f, "r")

            # Open this file part (magnitudes).
            this_f = f"{self.data_dir}/ivol_{i}_mags/magnitudes_sdss.hdf5"
            assert os.path.isfile(this_f), f"File {this_f} doesn't exist"
            f_mag_sdss = h5py.File(this_f, "r")
            this_f = f"{self.data_dir}/ivol_{i}_mags/magnitudes_2mass.hdf5"
            assert os.path.isfile(this_f), f"File {this_f} doesn't exist"
            f_mag_2mass = h5py.File(this_f, "r")

            # Loop over each attribute we are loading.
            for att in what_to_load:
                for F in [f, f_mag_sdss, f_mag_2mass]:
                    if att in F['Output001']:
                        assert len(F[f'Output001/{att}'].shape) == 1, "Bad shape"
                        if att in self.data.keys():
                            self.data[att] =\
                                np.concatenate((self.data[att],F[f'Output001/{att}'][...]))
                        else:
                            self.data[att] = F[f'Output001/{att}'][...]

            # Close files.
            f.close()
            f_mag_sdss.close()
            f_mag_2mass.close()

        # Remove h-factor from properties.
        self.convert_out_h()

        # Compute TrackId to match to HBT catalogue.
        if 'SubhaloID' in what_to_load and 'SubhaloSnapNum' in what_to_load:
            self.data['TrackId'] = np.zeros_like(self.data['SubhaloID'])
            self.data['TrackId'][:] = self.data['SubhaloID'] - 1e12*self.data['SubhaloSnapNum']

        # Total stellar mass.
        if 'mstars_disk' in what_to_load and 'mstars_bulge' in what_to_load:
            self.data['mstars'] = self.data['mstars_disk'] + self.data['mstars_bulge']

    def gather_galaxies(self):
        """ Reduce everything to rank 0. """

        if self.comm_size > 1:
            for att in self.data.keys():
                self.data[att] = self.comm.gather(self.data[att])

            if self.comm_rank == 0:
                for att in self.data.keys():
                    self.data[att] = np.concatenate(self.data[att])

    def convert_out_h(self, h=0.6777):
        """ Remove h-factor from loaded attributes. """
        facs = {"M_SMBH": 1./h, "mstars_bulge": 1./h, "mstars_disk": 1./h, "mhhalo": 1./h,
                "SubhaloID": None, "xgal": 1./h, "ygal": 1./h, "mhalo": 1./h,
                "zgal": 1./h, "vxgal": None, "vygal": None, "vzgal": None,
                "SubhaloSnapNum": None, "type": None}

        for att in self.data.keys():
            if "mag_" in att:
                self.data[att] += 5 * np.log10(h)
            elif att not in facs.keys():
                print(f"Warning: no h-factor conversion listed for {att}")
            else:
                if facs[att] is not None: self.data[att] *= facs[att]

    def link_sibelius(self, compute_distance=False, compute_ra_dec=False,
            compute_velocity=False, compute_galactic=False, compute_apparent_mag=False,
            compute_extra_coordinates=False, compute_extra_objects=False):
        """ Compute some properties of the subhaloes specific to Sibelius. """

        from sibelius_functions import compute_sibelius_properties

        compute_sibelius_properties(self.data, 'galform', compute_distance, compute_ra_dec,
                compute_velocity, compute_galactic, compute_apparent_mag,
                compute_extra_coordinates, compute_extra_objects)

    def mask_arrays(self, mask):
        """ Mask all fields by a given mask. """

        for att in self.data.keys():
            self.data[att] = self.data[att][mask]

if __name__ == '__main__':
    from mpi4py import MPI

    # MPI stuff.
    comm = MPI.COMM_WORLD

    d = "/cosma7/data/dp004/jch/SibeliusOutput/Sibelius_200Mpc_1/Galform/models/Lacey16/output/"
    x = read_galform(d, 10, comm=comm)

    #x.load_galaxies(["M_SMBH", "mstars_bulge", "SubhaloID", "xgal", "ygal", "zgal",
    #    "vxgal", "vygal", "vzgal", "mag_SDSS-g_o_bulge", "SubhaloSnapNum"])
    #x.link_sibelius(compute_distance=True, compute_ra_dec=True, compute_galactic=True,
    #        compute_apparent_mag=True, compute_velocity=True)

    #print(np.sort(x.data['TrackId']))

    x.load_galaxies(["type"])
    mask = np.where(x.data['type'] == 2)
    print(len(mask[0]))
    #print(x.data['xgal'] - 499.343)
    #print(x.data['ygal'] - 504.507)
    #print(x.data['zgal'] - 497.311)
    #ra = x.data['ra_rad'] + np.pi
    #dec = x.data['dec_rad'] + np.pi/2.
    #r = x.data['d_mw']

    #x = r*np.cos(ra)*np.sin(dec)
    #y = r*np.sin(ra)*np.sin(dec)
    #z = r*np.cos(dec)
    #print(x,y,z)
