import numpy as np
import h5py
from scipy.fftpack import dct, idct


class Sparsifier():

    def set_data(self, data):
        """ Assigns the data as an attribute. data can either be a numpy ndarray
        or an h5py Dataset object. Sets the following attributes:
        
            self.X                 array or h5py Dataset, NxP
            self.N                 number of rows of X (number of datapoints)
            self.P                 number of cols of X (latent dimension)
            self.gamma             compression factor (recomputed)
            self.M                 size of reduced latent dimension
        """

        if type(data) is h5py._hl.dataset.Dataset:
            X_type = 'hdf5'
        elif type(data) is np.ndarray:
            X_type = 'array'
        else:
            raise Exception('Data must either be an hdf5 dataset or a numpy array.')

        self.N, self.P = data.shape
        self.X = data

        if self.verbose:
            print('Data assigned as {}, '.format(X_type) + 
                'and is {} '.format(self.N) + 'by {} '.format(self.P) +
                '(data points by latent dimension).')

        # compute compression factor
        # if gamma is given as a float, find M
        if type(self.gamma) is float:
            self.M = int(np.floor(self.P * self.gamma))
        # if gamma is given as an integer, find gamma
        if type(self.gamma) is int:
            self.M = self.gamma
        # overwrite gamma (either because it was an int or to account for
        # rounding from floor function above)
        self.gamma = self.M/self.P
        if self.verbose:
            print('Latent dimension will be reduced to {} from'.format(self.M),
                '{} for a compression factor of {}.'.format(self.P, self.gamma))

    # ROS functions
    def set_ROS(self):
        """ Assigns the ROS and indices."""
        if self.use_ROS:
            # if we're told to compute it, do so
            if self.compute_ROS == True:
                HDX, D_indices = self.ROS(self.X[:])
                # ... but write it if we're allowed to so
                if self.write_permission == True:
                    self.write_ROS(self.fROS, HDX, D_indices)
            # otherwise load it
            elif self.fROS != None:
                HDX, D_indices = self.read_ROS(self.fROS)

        else:
            # if we're not using the ROS just set HDX to be X
            HDX = self.X[:].astype(float)
            D_indices = []
        self.HDX, self.D_indices = HDX, D_indices


    def ROS(self, X_dense):
        """ Apply the random orthogonal system transform to the full dataset, i.e.,
        compute HDX. D is diagonal +=1 so we just flip column signs, and for now H 
        is a discrete cosine transform."""
        if self.verbose:
            print('Computing ROS.')
        D_indices = [i for i in range(self.P) if np.random.choice([0,1])]
        X_dense[:,D_indices] *= -1
        X_dense = dct(X_dense, norm = 'ortho', axis = 1, overwrite_x = False) 
        return [X_dense, D_indices]

    def read_ROS(self, fROS):
        HDX = fROS['HDX']
        D_indices = fROS['D_indices']
        return [HDX, D_indices]

    def write_ROS(self, fROS, HDX, D_indices):
        """ Writes HDX and D_indices to file fROS (D_indices are needed to
        invert the transformation)."""
        
        if self.write_permission == False:
            raise Exception('Trying to write the ROS transform to disk but' + 
                             'write_permission is False')
        if 'HDX' in fROS:
            if self.verbose:
                print('Deleting existing ROS dataset in hdf5 file {}'.format(fROS))
            del fROS['HDX']
        if 'D_indices' in self.fROS:
            if self.verbose:
                print('Deleting existing D_indices dataset in hdf5 file {}'.format(fROS))
            del fROS['D_indices']
        if self.verbose:
            print('Writing ROS and D_indices to hdf5 file {}'.format(fROS))
        fROS.create_dataset('HDX', data = HDX, dtype = 'd')
        fROS.create_dataset('D_indices', data = D_indices, dtype = 'int')

    def __init__(self, gamma, verbose, fROS, write_permission,
                 use_ROS, compute_ROS):

        # assign constants
        self.gamma = gamma
        self.verbose = verbose
        self.fROS = fROS
        self.write_permission = write_permission
        self.use_ROS = use_ROS
        self.compute_ROS = compute_ROS


