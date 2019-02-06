import numpy as np

class DataGenerator():

    def get_centered_datapoint(self,i,k):
        """ Compute R_i^T (HDx_i - U_k). Helper for some other nasty 
        computations. """
        return self.RHDX[i] - self.U[k][self.mask[i]]

    def __init__(self):
        """ I think means and covariances correspond to sparsified ones. 
        """

        self.X = np.array([
            [ -8.85515933,  -5.34190685,   0.70918525,   3.6793991 , -6.16516529],
            [-11.79298152,  -0.31984685,   0.60067304,  -0.77468044, -2.80398769],
            [-11.4518398 ,  -4.4262754 ,   0.26197166,   4.07524417, -1.25984407],
            [ -6.09997943,  -2.68852669,   4.05670485,   2.05350523, 0.94251641]],
            dtype = np.float64)
        
        self.D_indices = np.array([0,3,4], dtype=int)

        self.HDX = np.array([[3, 1, 9, 2, 8],
                           [7, 5, 7, 4, 3],
                           [2, 6, 8, 4, 7],
                           [2, 4, 1, 3, 6]], dtype=np.float64)

        self.RRTHDX = np.array([[0, 1, 9, 0, 8],
                              [0, 0, 7, 4, 3],
                              [2, 0, 8, 0, 7],
                              [0, 4, 1, 3, 0]], dtype=np.float64)

        self.RHDX = np.array([[1, 9, 8],
                            [7, 4, 3],
                            [2, 8, 7],
                            [4, 1, 3]], dtype = np.float64)

        self.mask = np.array([[1, 2, 4],
                              [2, 3, 4],
                              [0, 2, 4],
                              [1, 2, 3]], dtype = np.int64)

        self.W = np.array([[0, 2, 7],
                           [4, 1, 8],
                           [1, 6, 4],
                           [3, 2, 8]], dtype = np.float64)

        self.U = np.array([[0, 6, 8, 3, 1],
                           [1, 3, 4, 7, 5],
                           [8, 9, 0, 4, 2]], dtype = np.float64)



        self.diagonal_covariances = np.array([[2, 3, 1, 1, 6],
                                              [7, 2, 1, 5, 4],
                                              [4, 2, 8, 9, 1]], dtype = np.float64)

        self.spherical_covariances = np.array([2,3,4], dtype = np.float64)

        self.N = 4
        self.Q = 3
        self.Qs = 1
        self.P = 5
        self.K = 3
        self.transform = 'dct'

        # hard-coded correct answers reused between tests

        self.correct_pairwise_l2_distances_with_self = \
            np.sqrt(
                np.array(
                    [[0     , 5/2*29, 5/2*2 , 5/2*73],
                     [5/2*29, 0     , 5/2*17, 5/2*37],
                     [5/2*2 , 5/2*17, 0     , 5*49  ],
                     [5/2*73, 5/2*37, 5*49  , 0     ]],
                     dtype = np.float64
                     )
                )

        self.correct_pairwise_l2_distances_with_full = \
            np.sqrt(5/3*
                np.array(
                    [[  75,  38, 181],
                     [   6,  22,  50],
                     [  40,  21, 125],
                     [  53,  26,  27]],
                     dtype = np.float64
                     )
                )

        self.correct_pairwise_mahalanobis_distances_spherical = \
            np.sqrt(5/3 * 
                np.array(
                    [[ 75/2,  38/3, 181/4],
                     [  6/2,  22/3,  50/4],
                     [ 40/2,  21/3, 125/4],
                     [ 53/2,  26/3,  27/4]],
                     dtype = np.float64
                     )
                )

        self.correct_pairwise_mahalanobis_distances_diagonal = \
            np.sqrt(5/3*
                np.array(
                    [[  17.5,  29.25, 78.125],
                     [   8/3,   11.8,  7.125],
                     [     8, 17+1/7,     42],
                     [ 151/3,   12.7, 12+53/72]],
                     dtype = np.float64
                     )
                )

        self.correct_logdet_spherical = \
            np.tile(
                    np.log(np.array([2**3, 3**3, 4**3], dtype = np.float64)),
                    (4,1))

        self.correct_logdet_diag = \
            np.log(np.array(
                [[  18,   8,  16],
                 [   6,  20,  72],
                 [  12,  28,  32],
                 [   3,  10, 144]], 
                dtype = np.float64
                )
            )



