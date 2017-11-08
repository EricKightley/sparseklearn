import numpy as np
from scipy import stats
from .sparsifier import Sparsifier

class KNeighborsClassifier(Sparsifier):


    def fit(self, X, y):
        self.initialize(X)
        self._y = y
        self.classes_ = list(set(y))
        

    def kneighbors(self, Xi, return_distances = True):
        Xicopy = np.copy(Xi)
        Xros = self.apply_ROS(Xi, self.D_indices)
        Xm = self.apply_mask(Xros, self.mask)

        dist = self.pairwise_distances(Xm, Y = self.HDX_sub)

        #neigh = dist.argsort(axis=1)#[:,:self.n_neighbors]
        #dist = np.sort(dist, axis=1)#[:,:self.n_neighbors]
        #if return_distances:
        #    d = np.array([dist[neighbors[:,k],k] for k in range(dist.shape[1])])
        #    return [d, neighbors.T]
        #else:
        #    return neighbors.T

        # apply HD to X
        X = self.ROS_test(Xicopy)
        
        test = np.array([X[n][self.mask[n]] for n in range(X.shape[0])] )

        kneighborsV = []
        kdistancesV = []
        for x in X:
            distances = np.zeros(self.N)

            for n in range(self.N):
                distances[n] = np.linalg.norm(self.HDX_sub[n] - x[self.mask[n]])

            neighbors = np.argsort(distances)
            #distances = distances[neighbors]

            kneighborsV.append(neighbors)#[:self.n_neighbors])
            kdistancesV.append(distances)#[:self.n_neighbors])

        kneighborsV = np.array(kneighborsV)
        kdistancesV = np.array(kdistancesV)
        print(np.all(kdistancesV==dist))
        import pdb; pdb.set_trace()

        return 0
        #if return_distances:
        #    return[kdistancesV, kneighborsV]
        #else:
        #    return kneighborsV

    def predict(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)
        y_pred = stats.mode(self._y[neigh_ind], axis = 1).mode.flatten()
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = np.sum(y_pred == y) / len(y_pred)
        return score

    def __init__(self, n_neighbors, **kwargs):

        super(KNeighborsClassifier, self).__init__(**kwargs)

        self.n_neighbors = n_neighbors
