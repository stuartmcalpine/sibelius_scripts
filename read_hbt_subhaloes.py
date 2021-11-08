import sys
import os
import h5py
import numpy as np

class HBT_data:

    def __init__(self, hbt_dir, comm=None, verbose=True):

        # Where is the HBT data.
        self.hbt_dir = hbt_dir

        # Default fields to load.
        self.what_to_load_default = ['ComovingMostBoundPosition',
                                     'Mbound',
                                     'PhysicalAverageVelocity',
                                     'TrackId',
                                     'Rank',
                                     'HostHaloId',
                                     'BoundR200CritComoving']
        
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
                    np.dtype('float32'): MPI.FLOAT,
                    np.dtype('float64'): MPI.DOUBLE,
                    np.dtype('int64'): MPI.LONG,
                    np.dtype('int32'): MPI.INT}

    def gather_haloes(self):
        """ Reduce everything to rank 0. """

        if self.comm_size > 1:
            # Make a new communicator for ranks with haloes.
            all_num_haloes = np.array(self.comm.allgather(self.num_haloes), dtype='i4')
            assert np.sum(all_num_haloes) > 0, 'Gathering no haloes'
            tmp_comm = self.comm.Split((0 if self.num_haloes > 0 else 1), self.comm_rank)

            if self.num_haloes > 0:

                # Number of haloes on each rank.
                rank_count = np.array(tmp_comm.allgather(self.num_haloes), dtype='i4')
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
                        recv_buf = np.empty(total_rank_count * reshape_size, dtype=dtype)
                    else:
                        recv_buf = None
        
                    tmp_comm.Gatherv(send_buf,
                        [recv_buf, myrankcount, rank_displ, self.dtype_to_mpi[dtype]])

                    # Master core.
                    if tmp_comm.rank == 0: 
                        self.data[att] = recv_buf
                        if reshape_size > 1:
                            self.data[att] = self.data[att].reshape(total_rank_count, reshape_size)

                # New number of haloes reduced to this rank.
                if tmp_comm.rank == 0:
                    self.num_haloes = total_rank_count
                else:
                    self.num_haloes = 0
                del recv_buf

            tmp_comm.Free()
            mask = np.where(np.array(self.comm.allgather(self.num_haloes)) > 0)[0]
            assert len(mask) == 1, 'Bad reduce'
            self.master_rank = mask[0]
            if self.comm_rank != self.master_rank: self.data = None

    def read_header(self, snapshotnumber):
        """ Read information from the header of file 0. """

        if self.comm_rank == self.master_rank:
            # Find HBT output.    
            first_file = self.hbt_dir + f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.0.hdf5"
            assert os.path.isfile(first_file), f'HBT file {first_file} not found'
        
            # Open first file
            f = h5py.File(first_file, 'r')
        
            # Read header.
            self.HEADER = {}
            for att in ['Cosmology/HubbleParam', 'NumberOfFiles', 'NumberOfSubhalosInAllFiles',
                    'SnapshotId', 'Cosmology/ScaleFactor']:
                self.HEADER[att] = f[att][0]

            f.close()
        else:
            self.HEADER = None

        if self.comm_size > 1: self.HEADER = self.comm.bcast(self.HEADER)

    def load_haloes(self, snapshotnumber, what_to_load=None):
        """ Load all subhaloes at this snapshotnumber. """
        # What are we loading?
        if what_to_load is None: what_to_load = self.what_to_load_default

        # Read header information from first file.
        self.read_header(snapshotnumber)
        num_file = self.HEADER['NumberOfFiles']

        # First index the files.
        self.num_haloes = 0
        for i in range(num_file):
            if i % self.comm_size != self.comm_rank: continue

            # Load this file part.
            this_file = self.hbt_dir+f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            assert os.path.isfile(this_file), f'HBT file {i} not found'
            
            if self.verbose: print(f"Indexing file {i+1} of {num_file}...")
            f = h5py.File(this_file, 'r')
            self.num_haloes += len(f['Subhalos'])
            f.close()

        # Loop over each file part and read the data.
        self.data = {}
        count = 0

        for i in range(num_file):
            if i % self.comm_size != self.comm_rank: continue

            # Load this file part.
            this_file = self.hbt_dir+f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            
            if self.verbose: print(f"Loading file {i+1} of {num_file}...")
            f = h5py.File(this_file, 'r')

            # Empty file?
            num_this_file = len(f['Subhalos'])
            if num_this_file == 0:
                f.close()
                continue

            # Loop over each attribute and save.
            for att in what_to_load:

                # Make array if it doesn't exist.
                if att not in self.data.keys():
                    dtype = getattr(np, f['Subhalos'][att].dtype.name)
                    if f['Subhalos'][att].ndim > 1:
                        self.data[att] = np.empty((self.num_haloes, f['Subhalos'][att].shape[1]),
                            dtype=dtype)
                    else:
                        self.data[att] = np.empty(self.num_haloes, dtype=dtype)

                # Populate array.
                self.data[att][count:count+num_this_file] = f['Subhalos'][att]

            count += num_this_file
            f.close()

    def load_subhalo_particles(self, snapshotnumber, trackidlist):
        # Read header information from first file.
        if not hasattr(self, 'HEADER'): self.read_header(snapshotnumber)

        # Loop over each file part and read the data.
        num_file = self.HEADER['NumberOfFiles']
        assert self.comm_size <= num_file, 'Too many ranks for files'

        data = {}
        for i in range(num_file):
            if i % self.comm_size != self.comm_rank: continue

            # Load this file part.
            this_file = self.hbt_dir+f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            assert os.path.isfile(this_file), f'HBT file {i} not found'
            
            if self.verbose: print(f"Loading file {i+1} of {num_file}...")
            f = h5py.File(this_file, 'r')

            # Empty file?
            if len(f['Subhalos']) == 0:
                f.close()
                continue

            # Loop over each attribute and save.
            for trackid in trackidlist:
                idx = np.where(f['Subhalos']['TrackId'][...] == trackid)[0]
                if len(idx) == 0: continue

                data[trackid] = f['SubhaloParticles'][idx][0]
                assert len(data[trackid]) == f['Subhalos']['Nbound'][idx], 'bad length'
            f.close()

        return data

    def mask_arrays(self, mask):
        """ Mask all fields by a given mask. """

        for att in self.data.keys():
            self.data[att] = self.data[att][mask]
            self.num_haloes = len(self.data[att])
        
    def link_sibelius(self, compute_distance=False, compute_ra_dec=False,
            compute_velocity=False, compute_galactic=False, compute_apparent_mag=False,
            compute_extra_coordinates=False, compute_extra_objects=False):
        """ Compute some properties of the subhaloes specific to Sibelius. """

        from sibelius_functions import compute_sibelius_properties

        if self.num_haloes > 0:
            compute_sibelius_properties(self.data, 'hbt', compute_distance, compute_ra_dec,
                    compute_velocity, compute_galactic, compute_apparent_mag,
                    compute_extra_coordinates, compute_extra_objects)

