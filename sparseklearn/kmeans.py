
import numpy as np
from .sparsifier import Sparsifier

class KMeans(Sparsifier) 
    
    def __init__(self, n_clusters = 8, init = 'k-means++', n_init = 10,
                 max_iter = 300, n_passes = 1, **kwargs):

        super(KMeans, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = 10
        self.max_iter = 300
        self.n_passes = 1
