.. polyfit documentation master file, created by
   sphinx-quickstart on Fri Nov 19 21:25:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to polyfit's documentation!
===================================
polyfit enables convenient multivariate sklearn-compatible polynomial regression. Compared to sklearn, the 
underlying convex optimization algorithms are sometimes more stable than the ones used by sklearn. 
Furthermore, polyfit enables shape constrained models to make the fit monotonic or convex for exmample.
However, these shape constraints are enforced by inequality constraints in the optimization problem which is numerically unstable.

Installation
===========================================
.. code:: bash

   pip install polyfit

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Example
   polyfit



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
