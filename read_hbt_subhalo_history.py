import sys
import os
import h5py
from scipy.spatial import distance
import numpy as np
from mpi4py import MPI

class HaloInfo:

    def __init__(self, comm_rank):

        self.found          = 0
        self.birth_snap     = 1e10
        self.num_snaps      = 0
        self.comm_rank      = comm_rank

    def assign_data(self, subhalo_data, snapshotnumber, a):
        """ Assign initial data from reference snapshot number. """
        self.data           = subhalo_data
        self.snapshotnumber = snapshotnumber
        self.a              = a
        self.num_snaps      = 1

    def get_birth_snap(self):
        """ Of these haloes, who was born first? """
        return np.min(self.data['SnapshotIndexOfBirth'])

    def add_data(self, new_data):
        """ Add data from another snapshot. """
        if self.num_snaps == 0:
            self.assign_data(new_data.data, new_data.snapshotnumber, new_data.a)
        else:
            self.data = np.hstack((self.data, new_data.data))
            self.snapshotnumber = np.concatenate((self.snapshotnumber, new_data.snapshotnumber))
            self.a = np.concatenate((self.a, new_data.a))
            self.num_snaps += 1

class subhalo_history:

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
            self.comm = comm
            self.comm_rank = comm.rank
            self.comm_size = comm.size

    def read_header(self, snapshotnumber):
        """ Read header information from the 0th HBT file. """

        if self.comm_rank == 0:
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

    def load_galaxy_history(self, trackid_list, snapshotnumber, sorted_by_trackid=False,
            up_to_max_snap=None):
        """ Return histories of given TrackIds. """

        # Read the header.
        self.read_header(snapshotnumber)

        # First find the galaxies in the reference snapshot.
        final_data = self.extract_history(trackid_list, snapshotnumber, sorted_by_trackid)

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
        assert found == len(trackid_list), f'Only found {found} of {len(trackid_list)}'

        # Loop over all the snapshots in the galaxy's history.
        if self.comm_rank == 0:
            print(f"Looping over {snapshotnumber-birth_snap+1} snapshots...")
        for tmp_snap in np.arange(birth_snap, snapshotnumber, 1):
            self.read_header(tmp_snap)
            this_data = self.extract_history(trackid_list, tmp_snap, sorted_by_trackid)

            if this_data.found > 0:
                final_data.add_data(this_data)

            if up_to_max_snap is not None:
                if tmp_snap >= up_to_max_snap: break

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
            for this_id in np.unique(final_data.data['TrackId']):
                results[this_id] = {}

                # Find this track id, sort by snapnum.
                mask = np.where(final_data.data['TrackId'] == this_id)
                sn = final_data.snapshotnumber[mask]
                mask2 = np.argsort(sn)

                for att in final_data.data.dtype.names:
                    results[this_id][att] = np.array(final_data.data[att][mask][mask2])
   
                results[this_id]['ScaleFactor'] = np.array(final_data.a[mask][mask2])
                results[this_id]['SnapshotNumber'] = \
                        np.array(final_data.snapshotnumber[mask][mask2])

            return results
        else:
            return None

    def extract_history(self, trackid_list, snapshotnumber, sorted_by_trackid):

        # Loop over each file part and read the data.
        num_file = self.HEADER['NumberOfFiles']
        trackid_list = np.sort(np.array(trackid_list))

        # Class to store halo info.
        thisHaloInfo = HaloInfo(self.comm_rank)
        num_total_found = 0

        # Loop over each file part.
        for i in range(num_file):

            if i % self.comm_size != self.comm_rank: continue

            # HBT file for this snapnum.
            this_file = self.hbt_dir + \
                    f"/{snapshotnumber:03d}/SubSnap_{snapshotnumber:03d}.{i}.hdf5"
            assert os.path.isfile(this_file), f'{this_file} not found'
            
            if self.verbose: print(f"[Rank {self.comm_rank}] Loading file {i+1} of {num_file}...")

            # Open file and look for tackid.
            f = h5py.File(this_file, 'r')

            # Should we even check this file?
            if sorted_by_trackid:
                if not (np.any(trackid_list >= f["Index/MinTrackIdThisFile"]) and \
                        np.any(trackid_list <= f["Index/MaxTrackIdThisFile"])):
                    f.close()
                    continue

            # Search.
            id_list = f['Subhalos']['TrackId'][...]

            if sorted_by_trackid:
                idx = np.searchsorted(id_list, trackid_list)
                idx[idx == len(id_list)] = len(id_list) - 1
                idx = np.array([x for i, x in enumerate(idx) if id_list[x] == trackid_list[i]])
            else:
                idx = np.where(np.in1d(id_list, trackid_list))[0]
            
            if len(idx) > 0:
                print(f"Rank {self.comm_rank} found {id_list[idx]} in file {this_file}",
                      f"(sn:{snapshotnumber})")
                thisHaloInfo.found += len(idx)

                # Extract.
                thisHaloInfo.assign_data(f['Subhalos'][idx], np.tile(snapshotnumber, len(idx)),
                    np.tile(self.HEADER['Cosmology/ScaleFactor'], len(idx)))

            f.close()

            # Do we need to keep going?
            if self.comm_size > 1:
                num_total_found = self.comm.allreduce(thisHaloInfo.found)
            else:
                num_total_found = thisHaloInfo.found
            if num_total_found == len(trackid_list): break

        return thisHaloInfo

if __name__ == '__main__':
    from mpi4py import MPI

    # MPI stuff.
    comm = MPI.COMM_WORLD

    #hbt_dir = '/cosma6/data/dp004/rttw52/swift_runs/runs/Sibelius/200Mpc/Sibelius_200Mpc_256/hbt/'
    hbt_dir = '/cosma6/data/dp004/rttw52/SibeliusOutput/Sibelius_200Mpc_1/hbt_refactored/'

    hbt = HBT_data(hbt_dir, comm=comm)
    
    x = hbt.load_galaxy_history([10566441,10566442], 199, sorted_by_trackid=True,
            up_to_max_snap=45)
    print(x)
    #hbt.load_galaxy_history([10566,10567], 199,  up_to_max_snap=75)
