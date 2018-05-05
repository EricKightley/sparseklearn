import unittest
import numpy as np

from sparseklearn import _l2_distance_both_masked

class DataGenerator():

    def __init__(self):

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

        self.Sigma = np.array([[2, 3, 1, 1, 6],
                               [7, 2, 1, 5, 4],
                               [4, 2, 8, 9, 1]], dtype = np.float64)

        self.N = 4
        self.Q = 3
        self.P = 5
        self.K = 3

class TestFastLAMethods(unittest.TestCase):

    def setUp(self):
        self.testdata = DataGenerator()

    def test__l2_distance_both_masked(self):
        """ Distance between RHDX[1] and RHDX[3]. """
        result = _l2_distance_both_masked(self.testdata.RHDX[1], 
                                         self.testdata.RHDX[3],
                                         self.testdata.mask[1],
                                         self.testdata.mask[3],
                                         self.testdata.Q,
                                         self.testdata.P)
        correct = np.sqrt(5/2. * 37)
        self.assertEqual(correct,result)
        #self.assertTrue(np.array_equal(self.testdata._U0_W0_Sig0_Pow1,result))

if __name__ == '__main__':
    unittest.main()
