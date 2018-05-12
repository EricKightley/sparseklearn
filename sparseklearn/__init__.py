from .sparsifier import Sparsifier
from .gmm import GaussianMixture
from .kmeans import KMeans

from .fastLA import dist_both_comp
from .fastLA import dist_one_comp_one_full
from .fastLA import pairwise_l2_distances_with_self
from .fastLA import pairwise_l2_distances_with_full
from .fastLA import mahalanobis_distance_spherical
from .fastLA import mahalanobis_distance_diagonal
from .fastLA import pairwise_mahalanobis_distances_spherical
from .fastLA import pairwise_mahalanobis_distances_diagonal

from .fastLA import update_weighted_first_moment
from .fastLA import update_weighted_first_moment_array
from .fastLA import compute_weighted_first_moment_array
from .fastLA import update_weighted_first_and_second_moment
from .fastLA import update_weighted_first_and_second_moment_array
from .fastLA import compute_weighted_first_and_second_moment_array
#from .auxutils import generate_mnist_dataset
#from .auxutils import load_mnist_dataset
#from .auxutils import write_mnist_dataset
#from .neighbors import KNeighborsClassifier
#from .mixture import GaussianMixture
