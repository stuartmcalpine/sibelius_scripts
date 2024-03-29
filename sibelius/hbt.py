import sys
import os
import h5py
import numpy as np
from scipy.spatial import distance
from sibelius.sibelius_functions import compute_sibelius_properties

class read_hbt_subhaloes:
    def __init__(self, hbt_dir, comm=None, verbose=True):
        """
        Object to read HBT output from Sibelius simulations.

        Reads from the SubSnap_***_*.hdf5 files. 

        When reading in MPI, each rank will read a subset of the total number
        of fileparts. Once read, you can reduce the arrays to rank=0 using
        self.gather_haloes().

        To compute Sibelius specific properties (RA/DEC etc), call
        self.link_sibelius().

        Parameters
        ----------
        hbt_dir : string
            Location of the HBT outputs
        comm : mpi4py communicator object (optional)
            Communicator for MPI reading
        verbose : bool
            More output?
        """

        # Where is the HBT data.
        self.hbt_dir = hbt_dir

        # Default fields to load.
        self.what_to_load_default = [
            "ComovingMostBoundPosition",
            "Mbound",
            "PhysicalAverageVelocity",
            "TrackId",
            "Rank",
            "HostHaloId",
            "BoundR200CritComoving",
        ]

        # Talkative?
        self.verbose = verbose

        # MPI stuff.
        self.master_rank = 0
        if comm is None:
            self.comm_rank = 0
            self.comm_size = 1
        else:
            from mpi4py import MPI

            self.comm = comm
            self.comm_rank = comm.rank
            self.comm_size = comm.size

            self.dtype_to_mpi = {
                np.dtype("float32"): MPI.FLOAT,
                np.dtype("float64"): MPI.DOUBLE,
                np.dtype("int64"): MPI.LONG,
                np.dtype("int32"): MPI.INT,
            }

    def gather_haloes(self):
        """Reduce everything to rank 0."""

        if self.comm_size > 1:
            # Make a new communicator for ranks with haloes.
            all_num_haloes = np.array(self.comm.allgather(self.num_haloes), dtype="i4")
            assert np.sum(all_num_haloes) > 0, "Gathering no haloes"
            tmp_comm = self.comm.Split(
                (0 if self.num_haloes > 0 else 1), self.comm_rank
            )

            if self.num_haloes > 0:

                # Number of haloes on each rank.
                rank_count = np.array(tmp_comm.allgather(self.num_haloes), dtype="i4")
                total_rank_count = np.sum(rank_count)

                # Loop over each loaded attribute and gather to master rank.
                for att in self.data.keys():
                    if self.data[att].ndim > 1:
                        reshape_size = self.data[att].shape[1]
                    else:
                        reshape_size = 1
                    myrankcount = rank_count * reshape_size
                    rank_displ = [sum(myrankcount[:p]) for p in range(tmp_comm.size)]
                    dtype = self.data[att].dtype
                    send_buf = np.ascontiguousarray(self.data[att])
                    if tmp_comm.rank == 0:
                        recv_buf = np.empty(
                            total_rank_count * reshape_size, dtype=dtype
                        )
                    else:
                        recv_buf = None

                    tmp_comm.Gatherv(
                        send_buf,
                        [recv_buf, myrankcount, rank_displ, self.dtype_to_mpi[dtype]],
                    )

                    # Master core.
                    if tmp_comm.rank == 0:
                        self.data[att] = recv_buf
                        if reshape_size > 1:
                            self.data[att] = self.data[att].reshape(
                                total_rank_count, reshape_size
                            )

                # New number of haloes reduced to this rank.
                if tmp_comm.rank == 0:
                    self.num_haloes = total_rank_count
                else:
                    self.num_haloes = 0
                del recv_buf

            tmp_comm.Free()
            mask = np.where(np.array(self.comm.allgather(self.num_haloes)) > 0)[0]
            assert len(mask) == 1, "Bad reduce"
            self.master_rank = mask[0]
            if self.comm_rank != self.master_rank:
                self.data = None

    def read_header(self, snapshotnumber):
        """
        Read information from the header of file 0.

        Parameters
        ----------
        snapshotnumber : int
            Snapshot we are reading from

        Attributes
        ----------
        self.HEADER : dict
            Contains some header information from HBT files
        """

        if self.comm_rank == self.master_rank:
            # Find HBT output.
            first_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.0.hdf5"
            )
            assert os.path.isfile(first_file), f"HBT file {first_file} not found"

            # Open first file
            f = h5py.File(first_file, "r")

            # Read header.
            self.HEADER = {}
            for att in [
                "Cosmology/HubbleParam",
                "NumberOfFiles",
                "NumberOfSubhalosInAllFiles",
                "SnapshotId",
                "Cosmology/ScaleFactor",
            ]:
                self.HEADER[att] = f[att][0]

            f.close()
        else:
            self.HEADER = None

        if self.comm_size > 1:
            self.HEADER = self.comm.bcast(self.HEADER)

    def load_haloes(self, snapshotnumber, what_to_load=None):
        """
        Load all subhaloes for a given snapshotnumber.

        Parameters
        ----------
        snapshotnumber : int
            Snapshot we are reading from
        what_to_load : list
            List of subhalo properties to load

        Attributes
        ----------
        self.data : dict
            Main data object containing subhalo catalog
        """

        # What are we loading?
        if what_to_load is None:
            what_to_load = self.what_to_load_default

        # Read header information from first file.
        self.read_header(snapshotnumber)
        num_file = self.HEADER["NumberOfFiles"]

        # First index the files.
        self.num_haloes = 0
        for i in range(num_file):
            if i % self.comm_size != self.comm_rank:
                continue

            # Load this file part.
            this_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            )
            assert os.path.isfile(this_file), f"HBT file {i} not found"

            if self.verbose:
                print(f"Indexing file {i+1} of {num_file}...")
            f = h5py.File(this_file, "r")
            self.num_haloes += len(f["Subhalos"])
            f.close()

        # Loop over each file part and read the data.
        self.data = {}
        count = 0

        for i in range(num_file):
            if i % self.comm_size != self.comm_rank:
                continue

            # Load this file part.
            this_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            )

            if self.verbose:
                print(f"Loading file {i+1} of {num_file}...")
            f = h5py.File(this_file, "r")

            # Empty file?
            num_this_file = len(f["Subhalos"])
            if num_this_file == 0:
                f.close()
                continue

            # Loop over each attribute and save.
            for att in what_to_load:

                # Make array if it doesn't exist.
                if att not in self.data.keys():
                    dtype = getattr(np, f["Subhalos"][att].dtype.name)
                    if f["Subhalos"][att].ndim > 1:
                        self.data[att] = np.empty(
                            (self.num_haloes, f["Subhalos"][att].shape[1]), dtype=dtype
                        )
                    else:
                        self.data[att] = np.empty(self.num_haloes, dtype=dtype)

                # Populate array.
                self.data[att][count : count + num_this_file] = f["Subhalos"][att]

            count += num_this_file
            f.close()

    def load_subhalo_particles(self, snapshotnumber, trackidlist):
        """
        HBT also stores the particle IDs bound to each subhalo.

        This loads the particles IDs for a given list of subhaloes.

        Parameters
        ----------
        snapshotnumber : int
            Snapshot we are reading from
        trackidlist : list
            List of TrackIds to load particles for

        Returns
        -------
        data : dict
            Dict of particle ids, indexed by the subhaloes TrackId
        """

        # Read header information from first file.
        if not hasattr(self, "HEADER"):
            self.read_header(snapshotnumber)

        # Loop over each file part and read the data.
        num_file = self.HEADER["NumberOfFiles"]
        assert self.comm_size <= num_file, "Too many ranks for files"

        data = {}
        for i in range(num_file):
            if i % self.comm_size != self.comm_rank:
                continue

            # Load this file part.
            this_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            )
            assert os.path.isfile(this_file), f"HBT file {i} not found"

            if self.verbose:
                print(f"Loading file {i+1} of {num_file}...")
            f = h5py.File(this_file, "r")

            # Empty file?
            if len(f["Subhalos"]) == 0:
                f.close()
                continue

            # Loop over each attribute and save.
            for trackid in trackidlist:
                idx = np.where(f["Subhalos"]["TrackId"][...] == trackid)[0]
                if len(idx) == 0:
                    continue

                data[trackid] = f["SubhaloParticles"][idx][0]
                assert len(data[trackid]) == f["Subhalos"]["Nbound"][idx], "bad length"
            f.close()

        return data

    def mask_arrays(self, mask):
        """Mask all fields by a given mask."""

        for att in self.data.keys():
            self.data[att] = self.data[att][mask]
        self.num_haloes = len(self.data[att])

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

        if self.num_haloes > 0:
            compute_sibelius_properties(
                self.data,
                "hbt",
                compute_distance,
                compute_ra_dec,
                compute_velocity,
                compute_galactic,
                compute_apparent_mag,
                observer,
            )


class _haloInfo:
    def __init__(self, comm_rank):

        self.found = 0
        self.birth_snap = 1e10
        self.num_snaps = 0
        self.comm_rank = comm_rank

    def assign_data(self, subhalo_data, snapshotnumber, a):
        """Assign initial data from reference snapshot number."""
        self.data = subhalo_data
        self.snapshotnumber = snapshotnumber
        self.a = a
        self.num_snaps = 1

    def get_birth_snap(self):
        """Of these haloes, who was born first?"""
        return np.min(self.data["SnapshotIndexOfBirth"])

    def add_data(self, new_data):
        """Add data from another snapshot."""
        if self.num_snaps == 0:
            self.assign_data(new_data.data, new_data.snapshotnumber, new_data.a)
        else:
            self.data = np.hstack((self.data, new_data.data))
            self.snapshotnumber = np.concatenate(
                (self.snapshotnumber, new_data.snapshotnumber)
            )
            self.a = np.concatenate((self.a, new_data.a))
            self.num_snaps += 1


class read_hbt_subhalo_history:
    def __init__(self, hbt_dir, comm=None, verbose=True):

        # Where is the HBT data.
        self.hbt_dir = hbt_dir

        # Talkative?
        self.verbose = verbose

        # MPI stuff.
        if comm is None:
            self.comm_rank = 0
            self.comm_size = 1
        else:
            from mpi4py import MPI

            self.comm = comm
            self.comm_rank = comm.rank
            self.comm_size = comm.size

    def read_header(self, snapshotnumber):
        """Read header information from the 0th HBT file."""

        if self.comm_rank == 0:
            # Find HBT output.
            first_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.0.hdf5"
            )
            assert os.path.isfile(first_file), f"HBT file {first_file} not found"

            # Open first file
            f = h5py.File(first_file, "r")

            # Read header.
            self.HEADER = {}
            for att in [
                "Cosmology/HubbleParam",
                "NumberOfFiles",
                "NumberOfSubhalosInAllFiles",
                "SnapshotId",
                "Cosmology/ScaleFactor",
            ]:
                self.HEADER[att] = f[att][0]

            f.close()
        else:
            self.HEADER = None

        if self.comm_size > 1:
            self.HEADER = self.comm.bcast(self.HEADER)

    def load_galaxy_history(
        self, trackid_list, snapshotnumber, sorted_by_trackid=False, up_to_max_snap=None
    ):
        """Return histories of given TrackIds."""

        # Read the header.
        self.read_header(snapshotnumber)

        # First find the galaxies in the reference snapshot.
        final_data = self.extract_history(
            trackid_list, snapshotnumber, sorted_by_trackid
        )

        if final_data.found > 0:
            birth_snap = final_data.get_birth_snap()
        else:
            birth_snap = 1e10

        # Find the youngest galaxy in the list.
        if self.comm_size > 1:
            found = self.comm.allreduce(final_data.found)
            birth_snap = self.comm.allreduce(birth_snap, op=MPI.MIN)
        else:
            found = final_data.found

        # Make sure we found all the passed trackids.
        assert found == len(trackid_list), f"Only found {found} of {len(trackid_list)}"

        # Loop over all the snapshots in the galaxy's history.
        if self.comm_rank == 0:
            print(f"Looping over {snapshotnumber-birth_snap+1} snapshots...")
        for tmp_snap in np.arange(birth_snap, snapshotnumber, 1):
            self.read_header(tmp_snap)
            this_data = self.extract_history(trackid_list, tmp_snap, sorted_by_trackid)

            if this_data.found > 0:
                final_data.add_data(this_data)

            if up_to_max_snap is not None:
                if tmp_snap >= up_to_max_snap:
                    break

        # Join up the results between cores.
        if self.comm_size > 1:
            all_data = self.comm.gather(final_data)

        # Concatenate arrays togther.
        if self.comm_rank == 0:

            # Add data from other ranks.
            if self.comm_size > 1:
                for tmp_data in all_data:
                    if tmp_data.comm_rank != 0 and tmp_data.num_snaps > 0:
                        final_data.add_data(tmp_data)

            results = {}
            for this_id in np.unique(final_data.data["TrackId"]):
                results[this_id] = {}

                # Find this track id, sort by snapnum.
                mask = np.where(final_data.data["TrackId"] == this_id)
                sn = final_data.snapshotnumber[mask]
                mask2 = np.argsort(sn)

                for att in final_data.data.dtype.names:
                    results[this_id][att] = np.array(final_data.data[att][mask][mask2])

                results[this_id]["ScaleFactor"] = np.array(final_data.a[mask][mask2])
                results[this_id]["SnapshotNumber"] = np.array(
                    final_data.snapshotnumber[mask][mask2]
                )

            return results
        else:
            return None

    def extract_history(self, trackid_list, snapshotnumber, sorted_by_trackid):

        # Loop over each file part and read the data.
        num_file = self.HEADER["NumberOfFiles"]
        trackid_list = np.sort(np.array(trackid_list))

        # Class to store halo info.
        thisHaloInfo = _haloInfo(self.comm_rank)
        num_total_found = 0

        # Loop over each file part.
        for i in range(num_file):

            if i % self.comm_size != self.comm_rank:
                continue

            # HBT file for this snapnum.
            this_file = (
                self.hbt_dir
                + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            )
            assert os.path.isfile(this_file), f"{this_file} not found"

            if self.verbose:
                print(f"[Rank {self.comm_rank}] Loading file {i+1} of {num_file}...")

            # Open file and look for tackid.
            f = h5py.File(this_file, "r")

            # Should we even check this file?
            if sorted_by_trackid:
                if not (
                    np.any(trackid_list >= f["Index/MinTrackIdThisFile"])
                    and np.any(trackid_list <= f["Index/MaxTrackIdThisFile"])
                ):
                    f.close()
                    continue

            # Search.
            id_list = f["Subhalos"]["TrackId"][...]

            if sorted_by_trackid:
                idx = np.searchsorted(id_list, trackid_list)
                idx[idx == len(id_list)] = len(id_list) - 1
                idx = np.array(
                    [x for i, x in enumerate(idx) if id_list[x] == trackid_list[i]]
                )
            else:
                idx = np.where(np.in1d(id_list, trackid_list))[0]

            if len(idx) > 0:
                print(
                    f"Rank {self.comm_rank} found {id_list[idx]} in file {this_file}",
                    f"(sn:{snapshotnumber})",
                )
                thisHaloInfo.found += len(idx)

                # Extract.
                thisHaloInfo.assign_data(
                    f["Subhalos"][idx],
                    np.tile(snapshotnumber, len(idx)),
                    np.tile(self.HEADER["Cosmology/ScaleFactor"], len(idx)),
                )

            f.close()

            # Do we need to keep going?
            # if self.comm_size > 1:
            #    num_total_found = self.comm.allreduce(thisHaloInfo.found)
            # else:
            #    num_total_found = thisHaloInfo.found
            # if num_total_found == len(trackid_list): break

        return thisHaloInfo
