Get Started
===========


You can install ``TARexp`` from `PyPi <https://pypi.org/project/tarexp/0.1.3/>`__ by running

.. code-block:: bash

    pip install tarexp


Or install it with the lastest version from GitHub

.. code-block:: bash

    pip install git+https://github.com/eugene-yang/tarexp.git

If you like to build it from source, please use

.. code-block:: bash

    git clone https://github.com/eugene-yang/tarexp.git
    cd tarexp
    python setup.py bdist_wheel
    pip install dist/*.whl


In Python, please use the following command to import both the main package and the components

.. code-block:: python

    import tarexp
    from tarexp import component