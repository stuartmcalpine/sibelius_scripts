import h5py

class read_velociraptor:

    def __init__(self, file_part, verbose=True):

        self.file_part = file_part
        self.verbose = verbose

        # Load header information.
        self.load_info()

    def load_info(self):
        """ Get information about this snapshot. """

        self.HEADER = {}
        with open(f"{self.file_part}.siminfo") as f:
            for line in f:
                att = line.split(":")[0].strip()
                value = float(line.split(":")[1].split('#')[0])
                self.HEADER[att] = value

    def load_galaxies(self, what_to_load):
        """ Load galaxy properties. """

        self.data = {}

        f = h5py.File(f"{self.file_part}.properties", 'r')

        for att in what_to_load:
            if att not in f:
                print(f"Warming, att={att} not found...skipping")
                continue
            if self.verbose: print(f"Loading {att}...", end="")
            self.data[att] =  f[att][...]
            if self.verbose: print(f"loaded {len(self.data[att])} galaxies.")

        f.close()

    def link_sibelius(self, compute_distance=False, compute_ra_dec=False,
            compute_velocity=False, compute_galactic=False, compute_apparent_mag=False,
            compute_extra_coordinates=False, compute_extra_objects=False, use_centre=False):
        """ Compute some properties of the galaxies specific to Sibelius. """

        from sibelius.sibelius_functions import compute_sibelius_properties

        compute_sibelius_properties(self.data, 'velociraptor', compute_distance, compute_ra_dec,
                compute_velocity, compute_galactic, compute_apparent_mag,
                compute_extra_coordinates, compute_extra_objects, use_centre)

if __name__ == '__main__':
    prop_file = '/cosma8/data/dp004/rttw52/DMO/Sibelius_200Mpc_1_FLAMINGO_9600/stf/0019/stf_0019'
    x = read_velociraptor(prop_file)
    #x.load_galaxies(['Mass_200crit'])

    #print(x.data)
