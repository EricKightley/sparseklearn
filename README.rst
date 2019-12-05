============
Sparseklearn
============

Dimensionality reduction for unsupervised machine learning.

Overview
--------

**Sparseklearn** is a Python package of machine learning algorithms
based on dimensionality reduction via random projections.
By working on compressed data,
**Sparseklearn** performs standard machine learning tasks
more efficiently and uses less memory. Its algorithms are all
*one-pass*, meaning that they only need to access the raw data
once, and are applicable to streaming data. **Sparseklearn** implements
algorithms described in our papers on sparsified `k-means
<https://arxiv.org/pdf/1511.00152.pdf>`_ and
`Gaussian mixtures
<https://arxiv.org/abs/1903.04056v2>`_.

Documentation
------------

Documentation is available at https://erickightley.github.io/sparseklearn/.

Installation
------------

Clone the repo, make a virtual environment, activate it, then:

.. code-block:: bash

    python setup.py build_ext --inplace
    pip install .

Test the installation by running the unit tests:

.. code-block:: bash

    pytest

Usage
-----

See :code:`examples/` for usage examples. You will need Jupyterlab:

.. code-block:: bash

    cd examples
    pip install -r requirements.txt
    jupyter lab
