import numpy as np

class TestData():

    def __init__(self):

        self.RHDX = np.array([[1, 9, 8],
                              [7, 4, 3],
                              [2, 8, 7],
                              [4, 1, 3]], dtype = np.float64)

        self.mask = np.array([[1, 2, 4],
                              [0, 2, 3],
                              [0, 2, 4],
                              [1, 2, 3]], dtype = np.int64)

        self.S = np.array([2,3], dtype = np.int64)

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

if __name__ == '__main__':
    testdata = TestData()


