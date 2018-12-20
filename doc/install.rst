Installation
============

The latest release can be installed from PyPi using::

    $ pip install topsbm

Install the development version from GitHub using::

    $ pip install https://github.com/Sydney-Informatics-Hub/topsbm/archive/master.zip

or by cloning the source code::

    $ git clone https://github.com/Sydney-Informatics-Hub/topsbm
    $ cd topsbm
    $ pip install .

Installing dependencies
.......................

topsbm requires `graph_tool <https://graph-tool.skewed.de/>`_ to already be
installed, as it cannot be installed with `pip`.

A simple way to install graph_tool and its dependencies is to use `conda
<https://repo.continuum.io/miniconda>`_::

    $ conda install -c conda-forge -c flyem-forge scikit-learn graph_tool pygobject cairo gtk3

or simply::

    $ git clone https://github.com/Sydney-Informatics-Hub/topsbm
    $ cd topsbm
    $ conda env create

Check your installation
.......................

Check the installation has worked with::

    $ python -m topsbm.check_install

or run the full test suite::

    $ pip install pytest
    $ pytest --pyargs topsbm

