import numpy as np
import h5py
import os


class read_galform:

    def __init__(
        self,
        data_dir,
        num_files,
        output_no,
        comm=None,
        verbose=True,
        only_single_ivol=None,
        to_convert_out_h=True
    ):
        """
        Object to read GALFORM output from Sibelius simulations.

        Reads from the "ivol_*" and "ivol_*_mags" subfolders (distinguished
        automatically from the passed parameters to load). 

        When reading in MPI, each rank will read a subset of the total number
        of ivols. Once read, you can reduce the arrays to rank=0 using
        self.gather_galaxies().

        To compute Sibelius specific properties (RA/DEC etc), call
        self.link_sibelius().

        Parameters
        ----------
        data_dir : string
            Directory containing the "ivol_*" subdirectories
        num_files : int
            The number of parts the catalog is split over
            i.e., the number of ivol folders
        output_no : int
            The output number (snapnum) to read (1 is z=0 usually)
        comm : mpi4py communicator object (optional)
            Communicator for MPI reading
        verbose : bool
            More output?
        only_single_ivol : int
            When wanting to read only a single ivol subpart
        to_convert_out_h : bool (default True)
            Convert out h-factors in the output
        """

        self.data_dir = data_dir
        self.num_files = num_files
        self.verbose = verbose
        self.output_no = output_no
        self.mag_list = ["2mass", "sdss"]
        self.only_single_ivol = only_single_ivol
        self.to_convert_out_h = to_convert_out_h

        # MPI stuff.
        if comm is None:
            self.comm_rank = 0
            self.comm_size = 1
        else:
            self.comm = comm
            self.comm_rank = comm.rank
            self.comm_size = comm.size

    def load_galaxies(self, what_to_load):
        """
        Load GALFORM galaxies.

        h-factors are converted out, therefore stored values are h-free.

        Parameters
        ----------
        what_to_load : list
            List of parameters to load

        Attributes
        ----------
        self.data : dict
            Main dictionary that holds the galaxy data
        self.HEADER : dict
            Some header information from run
        """

        assert type(what_to_load) == list, "what_to_load not a list"
        self.data = {}

        # Header.
        if self.comm_rank == 0:
            self.HEADER = {}
            fname = f"{self.data_dir}/ivol_0/galaxies.hdf5"
            f = h5py.File(fname, "r")
            assert f"Output{self.output_no:03d}" in f, "Output does not exist"
            self.HEADER["Redshift"] = float(
                f[f"Output{self.output_no:03d}/redshift"][...]
            )
            f.close()
        else:
            self.HEADER = None
        if self.comm_size > 1:
            self.HEADER = self.comm.bcast(self.HEADER)

        # What files to loop over.
        if self.only_single_ivol is not None:
            ivol_list = [self.only_single_ivol]
        else:
            ivol_list = range(self.num_files)

        # Loop over each file part.
        for i in ivol_list:
            if i % self.comm_size != self.comm_rank:
                continue

            # Open this file part (galaxies).
            f_list = []
            this_f = f"{self.data_dir}/ivol_{i}/galaxies.hdf5"
            assert os.path.isfile(this_f), f"File {this_f} doesn't exist"
            if self.verbose:
                print(f"[Rank {self.comm_rank}] Loading file {i}...")
            f_list.append(h5py.File(this_f, "r"))

            # Open this file part (magnitudes).
            for this_mag in self.mag_list:
                this_f = f"{self.data_dir}/ivol_{i}_mags/magnitudes_{this_mag}.hdf5"
                if os.path.isfile(this_f):
                    f_list.append(h5py.File(this_f, "r"))

            # Loop over each attribute we are loading.
            for att in what_to_load:
                for F in f_list:
                    if att in F[f"Output{self.output_no:03d}"]:
                        assert (
                            len(F[f"Output{self.output_no:03d}/{att}"].shape) == 1
                        ), "Bad shape"
                        if att in self.data.keys():
                            self.data[att] = np.concatenate(
                                (
                                    self.data[att],
                                    F[f"Output{self.output_no:03d}/{att}"][...],
                                )
                            )
                        else:
                            self.data[att] = F[f"Output{self.output_no:03d}/{att}"][...]

            # Close files.
            for f in f_list:
                f.close()

        # Remove h-factor from properties.
        if self.to_convert_out_h:
            self._convert_out_h()

        # Compute TrackId to match to HBT catalogue.
        if "SubhaloID" in what_to_load and "SubhaloSnapNum" in what_to_load:
            self.data["TrackId"] = np.zeros_like(self.data["SubhaloID"])
            self.data["TrackId"][:] = (
                self.data["SubhaloID"] - 1e12 * self.data["SubhaloSnapNum"]
            )

        # Total stellar mass.
        if "mstars_disk" in what_to_load and "mstars_bulge" in what_to_load:
            self.data["mstars"] = self.data["mstars_disk"] + self.data["mstars_bulge"]

    def gather_galaxies(self):
        """Reduce all arrays in self.data to rank 0."""

        if self.comm_size > 1:
            for att in self.data.keys():
                self.data[att] = self.comm.gather(self.data[att])

            if self.comm_rank == 0:
                for att in self.data.keys():
                    self.data[att] = np.concatenate(self.data[att])

    def _convert_out_h(self, h=0.6777):
        """
        The GALFORM output is in units of h.
        
        This removes the h-factor for each loaded attribute.
        
        If there is no conversion factor listed, it will still load, but you
        will get a warning.

        Parameters
        ----------
        h : float
            Hubble param
        """

        facs = {
            "M_SMBH": 1.0 / h,
            "mstars_bulge": 1.0 / h,
            "mstars_disk": 1.0 / h,
            "mhhalo": 1.0 / h,
            "SubhaloID": None,
            "xgal": 1.0 / h,
            "ygal": 1.0 / h,
            "mhalo": 1.0 / h,
            "zgal": 1.0 / h,
            "vxgal": None,
            "vygal": None,
            "vzgal": None,
            "SubhaloSnapNum": None,
            "type": None,
            "mstardot": 1.0 / h,
            "mstardot_average": 1.0 / h,
            "mstardot_burst": 1.0 / h,
        }

        for att in self.data.keys():
            if "mag_" in att:
                self.data[att] += 5 * np.log10(h)
            elif att not in facs.keys():
                print(f"Warning: no h-factor conversion listed for {att}")
            else:
                if facs[att] is not None:
                    self.data[att] *= facs[att]

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
        Compute some properties of the subhaloes specific to Sibelius.
        
        See sibelius_functions.py for options.
        """

        from sibelius.sibelius_functions import compute_sibelius_properties

        if int(self.output_no) != 1:
            print(
                "Warning: link_sibelius functions are assumed for the z=0 snapshot",
                f"this is the z={self.HEADER['Redshift']:.3f} snapshot",
            )
        compute_sibelius_properties(
            self.data,
            "galform",
            compute_distance,
            compute_ra_dec,
            compute_velocity,
            compute_galactic,
            compute_apparent_mag,
            observer,
        )

    def mask_arrays(self, mask):
        """Mask all fields by a given mask."""

        for att in self.data.keys():
            self.data[att] = self.data[att][mask]
