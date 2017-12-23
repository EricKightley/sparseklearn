import numpy as np
import h5py

from sparseklearn import Sparsifier
from sparseklearn import generate_mnist_dataset, load_mnist_dataset
from sparseklearn import KNeighborsClassifier
from sparseklearn import KMeans
from sparseklearn import GaussianMixture
from sklearn.cluster import KMeans as KMeansDefault

np.random.seed(17)



f = h5py.File('/home/eric/kmeansdata/mnistreduced.hdf5','r')
fROS = h5py.File('/home/eric/kmeansdata/fROS_temp.hdf5','a')

ntr = 1000
#nsmall = 100
n_train = {'0': ntr, '3' : ntr, '9' : ntr}
nte = 100
n_test = {'0': nte, '3' : nte, '9' : nte}

X_train, y_train, X_test, y_test =  generate_mnist_dataset(f, n_train, n_test)


#s = Sparsifier(gamma = .03, alpha = 0.01)
#s.fit_sparsifier(X_train)


gmm = GaussianMixture(gamma = 1.0, n_components = 3, covariance_type = 'diag',
                      init_params = 'random', normalize = False, max_iter = 2, use_ROS = False)
gmm.fit(X_train)
np.save('/home/eric/Dropbox/EricStephenShare/sparseklearn/plots/gmm_means.npy', gmm.means_)


#from sklearn.mixture import GaussianMixture as sklearnGaussianMixture
#clf = sklearnGaussianMixture(n_components=3, covariance_type='diag')
#clf.fit(X_train)

#np.save('/home/eric/Dropbox/EricStephenShare/sparseklearn/plots/sklearn_gmm_means.npy', clf.means_)
#d = gmm._estimate_gaussian_prob_diag(gmm.means_, np.random.rand(gmm.n_components,gmm.P))

#X = np.random.rand(10,5)
#gmm = GaussianMixture(gamma = 1.0, n_components = 2, covariance_type = 'diag', use_ROS = False)
#gmm.fit(X)
#means = np.zeros((2,gmm.P))
#means[1] = 1
#sigmas = .7*np.ones((2,gmm.P))
#f = gmm._estimate_gaussian_prob(means,sigmas)
#resp = gmm._estimate_resp(f, np.ones(gmm.n_components))
#f,resp = gmm._e_step(means,sigmas,gmm.weights_)
#gmm._m_step(f,gmm.HDX_sub)

"""
correct = np.zeros((gmm.N, gmm.n_components))
for n in range(gmm.N):
    for k in range(gmm.n_components):
        for d in range(gmm.P):
            correct[n,k] += (X[n,d] - means[k,d])**2/sigmas[k,d]
correct = np.sqrt(correct)

print(np.max(np.abs(correct - dist)))
"""

#s = Sparsifier(gamma = .03, alpha = 0.01)
#s.fit_sparsifier(X_train)

#knn = KNeighborsClassifier(gamma = 24, n_neighbors = 3, verbose = True)
#knn.fit(X_train, y_train)
#score = knn.score(X_test, y_test)
#print(score)


#kmc = KMeans(gamma = 0.03, alpha = 0.5, n_clusters = 3)
#kmc.fit(X_train)

#spa = Sparsifier(gamma=12, alpha = 4, verbose = True)
#spa.fit_sparsifier(X_train)


#kmc = KMeansDefault(n_clusters = 3, init='k-means++', n_init = 100)
#kmc.fit(X_train)

#n_per_cluster = np.zeros(kmc.n_clusters, dtype = int)
#
#for k in range(kmc.n_clusters):
#    n_per_cluster[k] = sum(kmc.labels_==k)
    
#np.save('/home/eric/Dropbox/EricStephenShare/sparseklearn/plots/counts.npy', n_per_cluster)




"""
ff = h5py.File('/home/eric/Dropbox/EricStephenShare/sparseklearn/tests/sample_mnist.h5py', 'a')
ff.create_dataset('X_train', data = X_train, dtype='i4', scaleoffset=0, 
        compression="gzip", compression_opts=9)
ff.create_dataset('X_test', data = X_test, dtype='i4', scaleoffset=0, 
        compression="gzip", compression_opts=9)
ff.create_dataset('y_train', data = y_train, dtype='i4', scaleoffset=0, 
        compression="gzip", compression_opts=9)
ff.create_dataset('y_test', data = y_test, dtype='i4', scaleoffset=0, 
        compression="gzip", compression_opts=9)
ff.close()


knn = KNeighborsClassifier(gamma = 5, verbose = False, fROS = None, 
                           write_permission = False, use_ROS = True, 
                           compute_ROS = True, dense_subsample = False,
                           constant_subsample = False,
                           n_neighbors = 3)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)

#T1 = knn.ROS_test(XT1)
#T2 = knn.apply_ROS(XT2, knn.D_indices)
X = np.random.rand(2,4)
Y = np.random.rand(2,4)

mask = np.array([[1,3],[0,2]])
#X_masked = knn.apply_mask(X,mask)
#Y_masked = knn.apply_mask(Y,mask)
D = np.array([-1,1])
X_ROSd = knn.apply_ROS(X, D)
Y_ROSd = knn.apply_ROS(Y, D)
X_ROSdmasked = knn.apply_mask(X_ROSd, mask)
d1 = knn.pairwise_distances(X_ROSdmasked,Y, mask = mask, D = D,
        transform_X = "", transform_Y = "RHD")

d2 = knn.pairwise_distances(X, Y, mask = mask, D = D, 
        transform_Y = "RHD", transform_X = "RHD")


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



