.. TARexp documentation master file, created by
   sphinx-quickstart on Sat Feb 26 22:18:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TARexp's documentation!
==================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   started
   API Documentation <modules>
   example
   references



``TARexp`` is an opensource Python framework for conducting TAR experiments with various
reference implementation to algorithms and methods that are commonly-used.

The experiments are fully reproducible and easy to conduct ablation studies. 
For studying components that do not change the selection process of the review documents, 
``TARexp`` supports replying TAR runs and experimenting these components offline. 

A major advance of  ``TARexp`` over previous TAR research software is the ability to declaratively specify TAR workflows.  
Users can create components defined using a standard interface and combine them with ``TARexp`` components in workflows of their design. 
This includes incorporating different simulations of human-in-the-loop reviewing, or even embedding in systems using actual human review 
(though we have not done the latter). 

Helper functions to support results analysis are also avaiable. 

Please visit our Google Colab Demo to check out the full example |Colab|.

.. |Colab| raw:: html

   <a href='https://colab.research.google.com/github/eugene-yang/tarexp/blob/main/examples/exp-demo.ipynb' target='_blank'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
