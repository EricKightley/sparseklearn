============
Sparseklearn
============

Dimensionality reduction for unsupervised machine learning.

.. inclusion-marker-do-not-remove

Overview
--------

**Sparseklearn** is a Python package of machine learning algorithms
based on dimensionality reduction via random projections.
By working on compressed data,
**Sparseklearn** performs standard machine learning tasks
more efficiently and uses less memory. Its algorithms are all
*one-pass*, meaning that they only need to access the raw data
once. **Sparseklearn** implements
algorithms described in our papers on sparsified `k-means and PCA
<https://arxiv.org/pdf/1511.00152.pdf>`_ and on
`Gaussian mixtures
<https://arxiv.org/abs/1903.04056v2>`_.

Documentation
-------------

Documentation is available at https://erickightley.github.io/sparseklearn/.

Installation
------------

It is highly recommended that you install this package in a
`virtual environment
<https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_.
With the virtual environment active, build the C extensions and install the
package:

.. code-block:: bash

    python setup.py build_ext --inplace
    pip install .

To test the installation, run the unit tests:

.. code-block:: bash

    pytest

Usage
-----

See :code:`examples/` for notebooks of usage examples. You will need Jupyterlab:

.. code-block:: bash

    cd examples
    pip install -r requirements.txt
    jupyter lab
