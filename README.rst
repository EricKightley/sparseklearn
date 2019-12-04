============
Sparseklearn
============

Dimensionality reduction for unsupervised machine learning.

Installation
------------
Clone the repo, make a virtual environment, activate it, then:

.. code-block:: bash

    $ python setup.py build_ext --inplace
    $ pip install .

Test the installation by running the unit tests:

.. code-block:: bash

    $ pytest

Overview
--------

**Sparseklearn** is a Python package of machine learning algorithms
based on dimensionality reduction via random projections.
By working on compressed data,
Sparseklearn performs standard machine learning tasks
more efficiently and uses less memory. Its algorithms are all
*one-pass*, meaning that they only need to access the raw data
once, and are applicable to streaming data.

A note on optimization
----------------------

Sparseklearn is being developed as a proof-of-concept for our work in
statistical learning and compressed sensing. It is currently in prototype stage
and has not yet been optimized. In particular,a lot of the computational demands
in Sparseklearn have been pushed to a preconditioning step, currently a discrete
cosine transform. This operation is fast and parallelizable, but it can still
be a bottleneck. We use scipy's dct function, but you may want to precompute
in a more efficient and distributed fashion. Sparseklearn can work with
preconditioned data - it does not need access to the orginal raw data.
