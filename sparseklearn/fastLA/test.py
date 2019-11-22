import os
import glob

path = os.path.abspath(__file__)
path = os.path.realpath(path)
path = os.path.dirname(path)
path = glob.glob(path + '/_fastLA*.so')[0]
