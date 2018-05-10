import numpy as np

class DataGenerator():

    def __init__(self):

        self.X = np.array([[3, 1, 9, 2, 8],
                           [7, 5, 7, 4, 3],
                           [2, 6, 8, 4, 7],
                           [2, 4, 1, 3, 6]], dtype=np.float64)

        self.RRTX = np.array([[0, 1, 9, 0, 8],
                              [0, 0, 7, 4, 3],
                              [2, 0, 8, 0, 7],
                              [0, 4, 1, 3, 0]], dtype=np.float64)

        self.RX = np.array([[1, 9, 8],
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
        self.P = 5
        self.K = 3

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






