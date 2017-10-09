.. neural-noise documentation master file, created by
   sphinx-quickstart on Mon Oct  9 10:44:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to neural-noise's documentation!
========================================


.. blockdiag examples: http://blockdiag.com/en/blockdiag/examples.html

.. blockdiag::
   :desctable:

   blockdiag {
      default_fontsize = 20;
      node_width = 200;
      node_height = 80;


      filter -> spike_detection -> spike_removal -> noise_imputation ->
      noise_discretization -> feature_extraction -> clustering;

      spike_removal -> noise_imputation [folded];
      feature_extraction -> clustering [folded];

      filter[label=Filter, description="Filter raw signals"]

      spike_detection[label="Spike detection", description="Detect spikes from filtered signals"]

      spike_removal[label="Spike removal", description="Remove spikes from the data to leave noise only"]

      noise_imputation[label="Noise Imputation", description="Impute noise in areas where spikes were removed"]

      noise_discretization[label="Noise discretization", description="Discretize noise"]

      feature_extraction[label="Feature extraction", description="Extract features from the noise chunks"]

      clustering[label="Clustering", description="Cluster noise chunks"]


   }


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
