
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

spr = Sparsifier(mask = mask, data_dim = P)
spr.fit_sparsifier(RHDX = RHDX)

#skm = KMeans(mask = mask, data_dim = P, n_clusters = 3)

skm = KMeans(n_clusters = 3, compression_target = 0.03, precond_D = precond_D)
skm.fit(HDX=HDX)
#localpath = '/home/eric/Dropbox/EricStephenShare/sparseklearn/plots/'

#np.save(localpath+'kmeans_means.npy', skm.cluster_centers_)
#np.save(localpath+'kmeans_counts.npy', skm.n_per_cluster)
