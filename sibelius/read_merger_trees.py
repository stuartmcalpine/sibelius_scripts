import numpy as np
import h5py

class read_merger_trees:

    def __init__(self, file_part, verbose=True):

        self.file_part = file_part
        self.verbose = verbose

    def load_galaxies(self, what_to_load):
        """ Load galaxy properties. """

        self.data = {}

        f = h5py.File(self.file_part, 'r')

        for att in what_to_load:
            if att in f['Subhalo']:
                if self.verbose: print(f"Loading {att}...", end="")
                self.data[att] =  f[f'Subhalo/{att}'][...]
                if self.verbose: print(f"loaded {len(self.data[att])} galaxies.")
            elif att in f['MergerTree']:
                if self.verbose: print(f"Loading {att}...", end="")
                self.data[att] =  f[f'MergerTree/{att}'][...]
                if self.verbose: print(f"loaded {len(self.data[att])} galaxies.")
            else:
                print(f"Warning, could not find {att}...skipping")
                continue

        f.close()

    def link_sibelius(self, compute_distance=False, compute_ra_dec=False,
            compute_velocity=False, compute_galactic=False, compute_apparent_mag=False,
            compute_extra_coordinates=False, compute_extra_objects=False, use_centre=False):
        """ Compute some properties of the galaxies specific to Sibelius. """

        from sibelius.sibelius_functions import compute_sibelius_properties

        compute_sibelius_properties(self.data, 'velociraptor', compute_distance, compute_ra_dec,
                compute_velocity, compute_galactic, compute_apparent_mag,
                compute_extra_coordinates, compute_extra_objects, use_centre)

    def find_history(self, galaxyid):
        mask = np.where(self.data['GalaxyID'] == galaxyid)[0]
        assert len(mask) == 1

        idx = np.where((self.data['GalaxyID'] >= galaxyid) &
                (self.data['GalaxyID'] <= self.data['TopLeafID'][mask]))[0]
        print(f"GalaxyID {galaxyid} has {len(idx)} entries on the main branch")

        return idx

if __name__ == '__main__':
    filepart = '/cosma6/data/dp004/jch/Sibelius/MultipleRealisations/DMO/Sibelius_200Mpc_1_FLAMINGO/merger_trees/Sibelius_200Mpc_1_FLAMINGO_9600/output/Velociraptor.0.hdf5'
    x = read_merger_trees(filepart)
    x.load_galaxies(['Mass_200crit','GalaxyID','SnapNum','Xc','Yc','Zc', 'TopLeafID'])
    x.link_sibelius(compute_distance=True, use_centre=True)
    x.find_history(433578)
