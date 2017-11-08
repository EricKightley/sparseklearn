import numpy as np
import h5py

from sparseklearn import Sparsifier
from sparseklearn import generate_mnist_dataset
from sparseklearn import KNeighborsClassifier

np.random.seed(17)



f = h5py.File('/home/eric/kmeansdata/mnistreduced.hdf5','r')
fROS = h5py.File('/home/eric/kmeansdata/fROS_temp.hdf5','a')

ntr = 500
nte = 50
n_train = {'0': ntr, '3' : ntr, '9' : ntr}
n_test = {'0': nte, '3' : nte, '9' : nte}

X_train, y_train, X_test, y_test =  generate_mnist_dataset(f, n_train, n_test)
XT1 = np.copy(X_test)
XT2 = np.copy(X_test)
XT3 = np.copy(X_test)
knn = KNeighborsClassifier(gamma = 12, verbose = False, fROS = None, 
                           write_permission = False, use_ROS = True, 
                           compute_ROS = True, dense_subsample = False,
                           constant_subsample = False,
                           n_neighbors = 3)
knn.fit(X_train, y_train)

#T1 = knn.ROS_test(XT1)
#T2 = knn.apply_ROS(XT2, knn.D_indices)
#X = np.random.rand(2,4)
#mask = np.array([[1,3],[0,2]])
#X_masked = knn.apply_mask(X,mask)
#distances = knn.pairwise_distances(X_test, X_train)

score = knn.score(X_test, y_test)
print(score)

"""

nruns = 30
n_neighbors = 3
gammaV = np.arange(2,28)
scores1 = []
scores2 = []
gout = []

for j in range(nruns):

    X_train, y_train, X_test, y_test =  generate_mnist_dataset(f, n_train, n_test)
    knn = KNeighborsClassifier(gamma = gammaV[0], verbose = False, fROS = fROS, 
                               write_permission = True, use_ROS = True, 
                               compute_ROS = True, dense_subsample = False,
                               n_neighbors = n_neighbors, constant_subsample = False)
    knn.initialize(X_train)
    for i in range(len(gammaV)):
        knn = KNeighborsClassifier(gamma = gammaV[i], verbose = False, fROS = fROS, 
                                   write_permission = False, use_ROS = True, 
                                   compute_ROS = False, dense_subsample = False,
                                   n_neighbors = n_neighbors, constant_subsample = True)
        knn.fit(X_train, y_train)
        scores1.append(knn.score(X_test, y_test))


        knn = KNeighborsClassifier(gamma = gammaV[i], verbose = False, fROS = fROS, 
                                   write_permission = False, use_ROS = True, 
                                   compute_ROS = False, dense_subsample = False,
                                   n_neighbors = n_neighbors, constant_subsample = False)
        knn.fit(X_train, y_train)
        scores2.append(knn.score(X_test, y_test))
        gout.append(gammaV[i])


gout = np.array(gout)
inds = np.argsort(gout)
gout = gout[inds]
scores1 = scores1[ind]
scotes2 = scores2[ind]

out = np.array([(x,y,z) for (x,y,z) in zip(gout, scores1, scores2)])

fp = 'comparison.csv'
np.savetxt(fp, out, delimiter = ',')


"""



