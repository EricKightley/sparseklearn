
import numpy as np
import h5py

from sparseklearn import Sparsifier
from sparseklearn import KMeans
from auxutils import generate_mnist_dataset


# set the random seed
rs = 22
np.random.seed(rs)

# load computed data
hdf5_file = h5py.File('/home/eric/kmeansdata/sample_mnist.hdf5','r')
X = hdf5_file['X'][:]
HDX = hdf5_file['HDX'][:]
RHDX = hdf5_file['RHDX'][:]
mask = hdf5_file['mask'][:]
precond_D = hdf5_file['precond_D'][:]
labels = hdf5_file['labels'][:]
# set P
P = 784



