import numpy as np
import h5py

from sparseklearn import Sparsifier

np.random.seed(42)

# Test data assignment

# ... as array
X_array = np.random.rand(500,300)
sparse = Sparsifier(gamma = 0.05, verbose = True, fROS = None, write_permission = False,
                    use_ROS = True, compute_ROS = True)
sparse.set_data(X_array)

# ... as hdf5 file
# TODO

# Test ROS computation
sparse.set_ROS()

# Test ROS read/write
# TODO






