import numpy as np
from scipy import stats
from .sparsifier import Sparsifier

class KNeighborsClassifier(Sparsifier):


    def fit(self, X, y):
        self.initialize(X)
        self._y = y
        self.classes_ = list(set(y))
        

    def kneighbors(self, X, return_distances = True):

        # apply HD to X
        X = self.ROS_test(X)

        kneighborsV = []
        kdistancesV = []
        for x in X:
            distances = np.zeros(self.N)

            for n in range(self.N):
                distances[n] = np.linalg.norm(self.HDX_sub[n] - x[self.mask[n]])

            neighbors = np.argsort(distances)
            distances = distances[neighbors]

            kneighborsV.append(neighbors[:self.n_neighbors])
            kdistancesV.append(distances[:self.n_neighbors])

        kneighborsV = np.array(kneighborsV)
        kdistancesV = np.array(kdistancesV)

        if return_distances:
            return[kdistancesV, kneighborsV]
        else:
            return kneighborsV

        
    def predict(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)
        y_pred = stats.mode(self._y[neigh_ind], axis = 1).mode.flatten()
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = np.sum(y_pred == y) / len(y_pred)
        return score

    def __init__(self, gamma, verbose, fROS, write_permission,
                 use_ROS, compute_ROS, dense_subsample, n_neighbors,
                 constant_subsample):

        Sparsifier.__init__(self, gamma, verbose, fROS, write_permission,
                             use_ROS, compute_ROS, dense_subsample,
                             constant_subsample)

        self.n_neighbors = n_neighbors
