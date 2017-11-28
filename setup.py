from setuptools import setup

setup(name='sparseklearn',
      version='0.1.3',
      url='http://github.com/EricKightley/sparseklearn',
      author='Eric Kightley',
      author_email='kightley.1@gmail.com',
      license='MIT',
      packages=['sparseklearn'],
      install_requires=[
          'scipy',
          'h5py',
          'numpy',
          'scikit-learn',
      ],
      zip_safe=False)
