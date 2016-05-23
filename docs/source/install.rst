Installation
============

Obtaining PyCLES
----------------

The simplest way to obtain the PyCLES source code is to clone it from GitHub_::

$ git clone https://github.com/pressel/pycles.git

Alternatively, we recommend forking the code directly from GitHub_  if you anticipate contributing to PyCLES. To fork the
repository, first create a GitHub_ account. Then visit the `PyCLES Github Page`_ and click the Fork button that appears on the
upper right side of the page. Once you have forked PyCLES, clone your fork.

.. _Github: http://www.github.com/
.. _`PyCLES Github Page`: https://github.com/pressel/pycles

Building PyCLES
---------------
PyCLES depends on several Python packages:

    1) numpy_ (tested with 1.9.1)
    2) scipy_ (tested with 0.16.0)
    3) matplotlib_ (tested with 1.4.3)
    4) Cython_ (tested with 0.23.1)
    5) mpi4py_ (tested with 1.3.1)
    6) netCDF4_ (tested with 1.1.9)

.. _numpy: http://www.numpy.org/
.. _scipy: http://www.scipy.org/
.. _matplotlib: http://matplotlib.org/
.. _Cython: https://pypi.python.org/pypi/Cython/
.. _mpi4py: https://pypi.python.org/pypi/mpi4py
.. _netCDF4: https://pypi.python.org/pypi/netCDF4/

Instructions for installation of these packages on Mac and Linux systems are given below.

Building on a Mac
+++++++++++++++++
PyCLES is developed on Macs, so installation on Mac systems is very straight forward.

Perhaps the simplest way to install PyCLES on a Mac is to use MacPorts_. Alternatively one could use Homebrew_, although the
developers have no experience doing so. If you have experience building the dependencies with Homebrew_ please
contribute your experience to this documentation!

.. _MacPorts: https://www.macports.org/
.. _Homebrew: http://brew.sh/

MacPorts_ makes the installation straight forward. We recommend using a version of Python installed using MacPorts. PyCLES
has been written to support both Python 2.7 and Python 3.x. In this documentation we will specifically
describe installation using Python 2.7, but installation using Python 3.x follows a similar progression. To install
Python 2.7 using Macports run::


$ [sudo] port install python27

Install NumPy::

$ [sudo] port install py27-numpy

Install SciPy::

$ [sudo] port install py27-scipy

Install mpi4py::

$ [sudo] port install py27-mpi4py

Install Cython::

$ [sudo] port install py27-cython

Install netCDF4::

$[sudo] port install py27-netcdf4

The above sequence of steps installs python to `/opt/local/bin` as `python2.7`, to make this version of python runnable
as `python` install `python_select`::

$[sudo] port install python_select

and then select python2.7 using::

$[sudo] port select --set python python27

Once all dependencies have been installed, move to the directory created when you cloned the PyCLES Github_ repository.
This probably requires something like::

$cd pycles

Once in the pycles directory run::

$ python generate_parameters.py

to generate parameters.pxi and ./Csrc/paramters.h. The source code can then be compiled by running setup.py ::

$ CC=mpicc python setup.py build_ext --inplace

.. note::
    In the above command, mpicc should be replaced by the name of your MPI C compiler.

Building on Linux
+++++++++++++++++

Building PyCLES on Linux systems involves installing the requires Python modules and their individual dependencies. The
python packages required by PyCLES are fairly standard and are likely pre-installed on most HPC systems. If the required
packages are not install we recommend consulting your system administrator to have them installed for you.

To install them on a non-HPC system we recommend using the package manager of your choice.

.. note::
    At ETH we have found it very useful on HPC systems to use a `Python Virtual Environment`_ to locally manage a Python
    environment without having administrator privileges.

.. _`Python Virtual Environment`: http://docs.python-guide.org/en/latest/dev/virtualenvs/

Once all dependencies have been installed, move to the directory created when you cloned the PyCLES Github_ repository.
This probably requires something like::

$cd pycles

Once in the pycles directory run::

$ python generate_parameters.py

to generate parameters.pxi and ./Csrc/paramters.h. The source code can then be compiled by running setup.py ::

Compile PyCLES::

$ CC=mpicc python setup.py build_ext --inplace

.. note::
    In the above command, mpicc should be replaced by the name of your MPI C compiler.

Site Specific Builds
--------------------

Euler @ ETH Zurich
++++++++++++++++++
PyCLES has been extensively tested and run on Euler using a Python Virtual Environment that has all of PyCLES's
dependencies pre-installed. To use the Python virtual environment add the following lines to your .bashrc file:

.. code-block:: bash

    module load new
    module load open_mpi
    module load python/2.7.6
    module load netcdf
    module load hdf5/1.8.12

    export PATH=/cluster/home/presselk/local2/bin:$PATH

From inside the PyCLES directory, parameters.pxi and ./Csrc/parameters.h can be generated by running::

$ python generate_parameters.py

Finally, PyCLES can be compiled by running::

$CC=mpicc python setup.py build_ext --inplace

