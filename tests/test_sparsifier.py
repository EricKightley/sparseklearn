import numpy as np
import h5py

from sparseklearn import Sparsifier
from sparseklearn import generate_mnist_dataset
from sparseklearn import KNeighborsClassifier

np.random.seed(42)



f = h5py.File('/home/eric/kmeansdata/mnistreduced.hdf5','r')
n_train = {'0': 1000, '3' : 1000, '9' : 1000}
n_test = {'0': 100, '3' : 100, '9' : 100}
X_train, y_train, X_test, y_test = generate_mnist_dataset(f, n_train, n_test)

knn = KNeighborsClassifier(gamma = 15, verbose = True, fROS = None, 
                           write_permission = False, use_ROS = True, 
                           compute_ROS = True, dense_subsample = False,
                           n_neighbors = 5)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)


#distances, neighbors = knn.kneighbors(X_test[0:2], n_neighbors = 2)




