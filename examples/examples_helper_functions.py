import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


face = misc.face(gray=True).astype(int)
fR,fC = face.shape
flatface = face.flatten()[np.newaxis,:]

def plot_mnist_means(kmc):
    f, (ax0, ax1, ax2) = plt.subplots(figsize=(10,32),ncols=3)
    axeslist = [ax0,ax1,ax2]
    for i in range(3):
        ax = axeslist[i]
        ax.set_title("count = {}".format(kmc.n_per_cluster[i]))
        ax.imshow(kmc.cluster_centers_[i].reshape(28,28),cmap='Greys')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.show()
