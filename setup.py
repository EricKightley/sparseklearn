from setuptools import setup
import subprocess

subprocess.call(['make', '-C', 'sparseklearn/source'])

setup(name='sparseklearn',
      version='0.1.4',
      url='http://github.com/EricKightley/sparseklearn',
      author='Eric Kightley',
      author_email='kightley.1@gmail.com',
      license='MIT',
      packages=['sparseklearn'],
      package_data={'sparseklearn': ['libauxiliary.so', 'libdistances.so', 'libmoments.so']},
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
          'scikit-learn',
      ],
      zip_safe=False)
