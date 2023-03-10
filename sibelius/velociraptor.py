import h5py
import os

from sibelius.sibelius_functions import compute_sibelius_properties


class read_velociraptor:
    """
    Read a Velociraptor STF data file.

    Exclude the final part of the filename (i.e, the ".properties").

    Example usage:
    --------------
        stf_path = "/my/stf/path/stf_1000"
        vel = read_velociraptor(stf_path)

    Note: does NOT work in MPI.
    """

    def __init__(self, file_part, verbose=True):
        """
        Parameters
        ----------
        file_part : string
            full path to STF file, or file part, we are reading
        verbose : bool (optional)
            true for more stdout
        """

        self.file_part = file_part
        self.verbose = verbose
        self.num_galaxies = 0

        # Load header information.
        self.load_info()

    def load_info(self):
        """
        Get some basic information about this STF output.

        Reads the information from the "siminfo" STF file.

        Returns
        -------
        None
            stores the information in a dict named "HEADER"
        """

        # Dict to store header information.
        self.HEADER = {}

        # Read the "siminfo" file and store results in dict.
        with open(f"{self.file_part}.siminfo") as f:
            for line in f:
                att = line.split(":")[0].strip()
                value = float(line.split(":")[1].split("#")[0])
                self.HEADER[att] = value

        # Print header info.
        if self.verbose:
            print(f"Header info for {self.file_part}")
            for att in self.HEADER.keys():
                print(f" - {att}: {self.HEADER[att]}")

    def load_galaxies(self, what_to_load):
        """
        Load the selected galaxy properties.

        Parameters
        ----------
        what_to_load : list of strings
            the parameters we want to load

        Returns
        -------
        None
            stores the galaxy data in a dict named "data"
        """

        assert len(what_to_load) > 0

        # Dict to store the data.
        self.data = {}

        fname = f"{self.file_part}.properties"
        if not os.path.isfile(fname):
            fname = f"{self.file_part}.properties.0"

        f = h5py.File(fname, "r")

        # Loop over each property to load and read.
        for att in what_to_load:
            if att not in f:
                print(f"Warming, att={att} not found...skipping")
                continue
            if self.verbose:
                print(f"Loading {att}...", end="")
            self.data[att] = f[att][...]
            if self.verbose:
                print(f"loaded {len(self.data[att])} galaxies.")

        self.num_galaxies = len(self.data[att])
        f.close()

    def link_sibelius(
        self,
        compute_distance=False,
        compute_ra_dec=False,
        compute_velocity=False,
        compute_galactic=False,
        compute_apparent_mag=False,
        observer="sibelius_dark_mw",
    ):
        """
        Compute some properties of the galaxies specific to Sibelius.
        
        See sibelius_functions.py for options.
        """
        if self.num_galaxies > 0:
            compute_sibelius_properties(
                self.data,
                "velociraptor",
                compute_distance,
                compute_ra_dec,
                compute_velocity,
                compute_galactic,
                compute_apparent_mag,
                observer,
            )
