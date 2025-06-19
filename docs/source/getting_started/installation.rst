============
Installation
============

Clone the repository and change directory:

.. code-block:: console

  $ git clone https://github.com/pnnl/irsa.git
  $ cd irsa/

Use `conda <https://www.anaconda.com/download/>`_ (or, more efficiently, `mamba <https://mamba.readthedocs.io/en/latest/>`_) to create a virtual environment with required dependencies.

On Mac or Linux:

.. code-block:: console
  
  $ conda env create -f envs/environment.yml
  or
  $ mamba env create -f envs/environment.yml

On CUDA appropriate systems:

.. code-block:: console
  
  $ conda env create -f envs/environment_cuda.yml
  or
  $ mamba env create -f envs/environment_cuda.yml

Activate the virtual environment:

.. code-block:: console
  
  $ conda activate irsa

Install IRSA using `pip <https://pypi.org/project/pip/>`_:

.. code-block:: console
  
  $ pip install -e .
