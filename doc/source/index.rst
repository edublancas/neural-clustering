.. neural-noise documentation master file, created by
   sphinx-quickstart on Mon Oct  9 10:44:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to neural-noise's documentation!
========================================

.. graphviz::

   digraph {
      node[shape=record]

      filter -> spike_detection -> spike_removal -> noise_imputation ->
      noise_discretization -> feature_extraction -> clustering

      filter[label=Filter]
      spike_detection[label="Spike detection"]
      spike_removal[label="Spike removal"]
      noise_discretization[label="Noise discretization"]
      noise_imputation[label="Noise Imputation"]
      feature_extraction[label="Feature extraction"]
      clustering[label="Clustering"]

   }


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
