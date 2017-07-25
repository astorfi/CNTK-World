=======================
Install CNTK Manually
=======================

.. _The Microsoft Cognitive Toolkit: https://docs.microsoft.com/en-us/cognitive-toolkit/
.. _Setup CNTK on your machine: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine
.. _Bazel Installation: https://bazel.build/versions/master/docs/install-ubuntu.html
.. _CUDA Installation: https://github.com/astorfi/CUDA-Installation
.. _NIDIA documentation: https://github.com/astorfi/CUDA-Installation



The installation is available at `The Microsoft Cognitive Toolkit`_ page. Manual installation is recommended because the user can build the desired CNTK binary for the specific architecture.
It enriches the CNTK with a better system compatibility and it will run much faster.
Manual installation is available at `Setup CNTK on your machine`_ link.
The official CNTK explanations are concise and to the point. However. few things might become important as we go through the installation. We try to project the step by step process to avoid any confusion. The following sections must be considered in the written order.

The assumption is that installing ``CNTK 2.0`` in the ``Ubuntu`` using ``GPU support`` is desired. ``Python2.7`` is chosen for installation.

.. **NOTE** Please refer to this youtube `link <youtube_>`_ for a visual explanation.

.. .. _youtube: https://www.youtube.com/watch?v=_3JFEPk4qQY&t=2s

.. _C++ Compiler: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux#c-compiler
.. _Open MPI Installation: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux#open-mpi

------------------------
Prepare the environment
------------------------

The following should be done in order:

    * CNTK system dependencies installation
    * CNTK Python dependencies installation
    * GPU prerequisites setup
    * `C++ Compiler`_
    * `Open MPI Installation`_


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
System Dependencies Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

  sudo apt-get install autoconf automake libtool curl make g++ unzip


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Dependencies Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For installation of the required dependencies, the following command must be executed in the terminal:

.. code:: bash

    sudo apt-get install python-numpy python-dev python-pip python-wheel python-virtualenv

~~~~~~~~~~~~~~~~~~~~~~~~
GPU Prerequisites Setup
~~~~~~~~~~~~~~~~~~~~~~~~

The following requirements must be satisfied:

    * NVIDIA's Cuda Toolkit and its associated drivers (version 8.0 is recommended). The installation is explained at `CUDA Installation`_.
    * The cuDNN library (version 5.1 or higher is recommended). Please refer to `NIDIA documentation`_ for further details.
    * Installing the ``libcupti-dev`` using the following command: ``sudo apt-get install libcupti-dev``

**The main goal is to make sure the latest NVIDIA driver is installed.**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creating a Virtual Environment (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume the installation of TensorFlow in a ``python virtual environment`` is desired. First, we need to create a directory to contain all the environments. It can be done by executing the following in the terminal:

.. code:: bash

    sudo mkdir ~/virtualenvs

Now by using the ``virtualenv`` command, the virtual environment can be created:

.. code:: bash

    sudo virtualenv --system-site-packages ~/virtualenvs/CNTK

or the following for python3:

.. code:: bash

    sudo virtualenv --p python3 ~/virtualenvs/CNTK

**Environment Activation**

Up to now, the virtual environment named *CNTK* has been created. For environment activation, the following must be done:

.. code:: bash

    source ~/virtualenvs/CNTK/bin/activate

However, the command is too verbose!

**Alias**

The solution is to use an alias to make life easy! Let's execute the following command:

.. code:: bash

    echo 'alias CNTK="source $HOME/virtualenvs/CNTK/bin/activate" ' >> ~/.bash_aliases
    bash

After running the previous command, please close and open terminal again. Now by running the following simple script, the tensorflow environment will be activated.

.. code:: shell

    CNTK

**check the ``~/.bash_aliases``**

To double check let's check the ``~/.bash_aliases`` from the terminal using the ``sudo gedit ~/.bash_aliases`` command. The file should contain the following script:

.. code:: shell

    alias CNTK="source $HO~/virtualenvs/CNTK/bin/activate"


**check the ``.bashrc``**

Also, let's check the ``.bashrc`` shell script using the ``sudo gedit ~/.bashrc`` command. It should contain the following:

.. code:: shell

    if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
    fi


~~~~~~~~~~~~~~~~~~~~~~~~~~
C++ Compiler Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The C++ compiler might be naively installed. In the Ubuntu, you can check it as follows:

.. code:: shell

    dpkg --list | grep compiler

Please refer to the `C++ Compiler`_ documentation.


~~~
MKL
~~~

Intel Math Kernel Library (Intel MKL) is the default CNTK math library is the.

**As Microsoft says**: *"You can NOT directly build CNTK using a regular
installed Intel MKL SDK, the build is configured to work with a custom
generated CNTK custom MKL library (This way you don't need to go through
the process of installing the complete Intel MKL SDK).*

The installation process is as follows:

* Create a directory to hold CNTK custom MKL:

  .. code:: shell

      sudo mkdir /usr/local/CNTKCustomMKL

.. _Cognitive Toolkit Custom MKL Package: https://www.microsoft.com/en-us/cognitive-toolkit/download-math-kernel-library/

* Download the required CNTK custom MKL from `Cognitive Toolkit Custom MKL Package`_ page.


* Unpack it in the created directory:

  * .. code:: shell

      sudo tar -xzf CNTKCustomMKL-Linux-3.tgz -C /usr/local/CNTKCustomMKL

For configuration of ``CNTK``, ``--with-mkl=<directory>`` option must be used. In
our case, ``--with-mkl=/usr/local/CNTKCustomMKL`` is the correct flag.

~~~~~~~~~~~~~~~~~~~~~~~~~~
Open MPI Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _Open MPI: https://www.open-mpi.org/

`Open MPI`_ is a **High Performance Message Passing Library**. It is an important part of manual installation of CNTK for having a better performance and make the most of it.

The procedure for Open MPI installation is as below:

* Getting the source of installation:

  * .. code:: shell

      wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz

* Unpack it:

  * .. code:: shell

     tar -xzvf ./openmpi-1.10.3.tar.gz cd openmpi-1.10.3

* Configuration:

  * .. code:: shell

      ./configure --prefix=/usr/local/mpi

* Build & Install:

  * .. code:: shell

     make -j all && sudo make install


* Add the environment variable to ``.bashrc`` profile:

  * .. code:: shell

     export PATH=/usr/local/mpi/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH


~~~~~~~~~~~~~~~~~~~~~~~~~~
Protobuf Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

In CNTK Protocol Buffers is used for serialization. It should be installed by the following procedure:


* Installing the required packages:

  * .. code:: shell

     sudo apt-get install autoconf automake libtool curl make g++ unzip


* Get the Protobuf from the source:

  * .. code:: shell

      wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz && tar -xzf v3.1.0.tar.gz


* Compiling Protobuf && Installation:

  * .. code:: shell

      cd protobuf-3.1.0 && ./autogen.sh && ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/protobuf-3.1.0 && make -j $(nproc) && sudo make install


~~~~~~~~~~~~~~~~~~~~~~~~~~
Zlib Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _zlib: http://zlib.net/

You can get the latest version from `zlib`_ offical website. Alternatively, it can be installed in Ubuntu using the following command:


.. code:: shell

    sudo apt-get install zlib1g-dev

~~~~~~~
LIBZIP
~~~~~~~

.. _LIBZIP: http://zlib.net/

`LIBZIP`_ is a C library for reading, creating, and modifying zip archives. It is recommended
to install ``LIBZIP`` from the source. The procedure is as follows:


* Get and unpack the source file:

  * .. code:: shell

        wget http://nih.at/libzip/libzip-1.1.2.tar.gz && tar -xzvf ./libzip-1.1.2.tar.gz



* Configuration & Installation:

  * .. code:: shell

      cd libzip-1.1.2 && ./configure && make -j all && sudo make install

Now the environment variable must be added to ``.bashrc`` profile:

.. code:: shell

    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boost Library Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boost Library is an important prerequisite for CNTK setup. The installation process is as follows:


* Installing dependencies:

  * .. code:: shell

      sudo apt-get install libbz2-dev && sudo apt-get install python-dev


* Getting the source files:

  * .. code:: shell

      wget -q -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf -


* Installation:

  * .. code:: shell

      cd boost_1_60_0 && ./bootstrap.sh --prefix=/usr/local/boost-1.60.0 && sudo ./b2 -d0 -j"$(nproc)" install


~~~~~~~~~~~~~~~~~~~~~~~~~~~
NCCL Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _NCCL library : https://github.com/NVIDIA/nccl

NVIDIA's `NCCL library`_ can be installed for optimized multi-GPU
communication on Linux which CNTK can take advantage from it.

Please follow build instructions as follows:

* Clone the NCCL repository:

  * .. code:: shell

      git clone https://github.com/NVIDIA/nccl.git $$ cd nccl


* Build $$ Test:

  * .. code:: shell

      make CUDA_HOME=<cuda install path> test

In which ``<cuda install path>`` is usually ``/usr/local/cuda``.


* Add to path:

  * .. code:: shell

      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib

* Build tests:

  * .. code:: shell

      ./build/test/single/all_reduce_test


You may get the error of ``Error: must specify at least data size in bytes!``. Then
run the following:

.. code:: shell

      ./build/test/single/all_reduce_test 10000000


**WARNING**: In configuration of CNTK, ``--with-nccl=<path>`` option must be used
to enable ``NVIDIA NCCL``. In our example ``$HOME/nccl/build`` in the ``path argument``.


~~~~~~~~~~~~~~~~~~
SWIG Installation
~~~~~~~~~~~~~~~~~~

SWIG is required if Python is desired to be the interface for CNTK. The process is as follows:

.. code:: shell

      sudo [CNTK clone root]/Tools/devInstall/Linux/install-swig.sh

This is expected to install SWIG in ``/usr/local/swig-3.0.10``.

**WARNING**: It is very important to use ``sudo`` for SWIG installation.


-----------------------
CNTK setup for Python
-----------------------

---------------------------------
build CNTK with Python support
---------------------------------

~~~~~~~~~~~~~~~~~
Build Python APIs
~~~~~~~~~~~~~~~~~

The step-by-step procedure is as fllows:

* Make sure ``SWIG`` is installed.
* Make sure Anaconda, Miniconda or any other environment (which contains conda environment) is installed.
* Create the conda environment as follows (for a Python X-based version in which X can be ``27``, ``34``, ``35``, ``36`` equivalent to ``2.7``, ``3.4``, ``3.5``, ``3.6``):

  * .. code:: shell

      conda env create --file [CNTK clone root]/Scripts/install/linux/conda-linux-cntk-pyX-environment.yml

* Now, since we have the environment, the packages can be updated to latest versions as below:

  * .. code:: shell

      conda env update --file [CNTK clone root]/Scripts/install/linux/conda-linux-cntk-pyX-environment.yml --name cntk-pyX

* Now, the conda environment can be activated as below:

  * .. code:: shell

      source activate cntk-pyX

**NOTE**: Remember to set ``X`` according to the desired version and existing files.

~~~~~~~~~~~~~~~~~~~~~~~
Before Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. _Multiverso: https://github.com/microsoft/multiverso

Parameter server is a framework which is of great importance in distributed machine learning.
Asynchronous parallel training with many workers is one of the key advantages. before
configuration of ``CNTK`` we are determined to build CNTK with `Multiverso`_ supported.
Multiverso is a parameter server framework developed by Microsoft Research Asia team. It enables the Asynchronous SGD.

The installation process is as follows:

* cd the root folder of CNTK.

* Clone the ``Multiverso`` code under the root folder of CNTK:

  * .. code:: shell

      git submodule update --init Source/Multiverso

* In ``CNTK`` configuration, use the ``--asgd=yes`` flag (Linux).

~~~~~~~~~~~~~~~~~~~~~~~
Building Python Package
~~~~~~~~~~~~~~~~~~~~~~~

Configuration is as follows as the user is the directory of ``CNTK clone root``.

.. code:: shell

    ./configure  --with-swig=/usr/local/swig-3.0.10 --with-py35-path=$HOME/anaconda/envs/cntk-py35 --with-nccl=$HOME/GITHUB/nccl/build --with-mkl=/usr/local/CNTKCustomMKL --asgd=yes


Now, the ``.whl`` file has been created. Installation of ``CNTK`` is as follows:

* cd to the folder that ``.whl`` file is located.

  * .. code:: shell

      cd [CNTK clone root]/python


* Activate virtual environment.

  * .. code:: shell

      source activate cntk-py35


* Install the created package using ``pip``.

  * .. code:: shell

      pip install file_name.whl

--------------------------
Validate the Installation
--------------------------

In the terminal, the following script must be run (``in the home directory``) correctly without any error and preferably any warning:

.. code:: bash

    python

    >> import cntk


--------------------------
Summary
--------------------------

In this tutorial, we described how to install CNTK from the source which has the
advantage of more compatibility with the system configuration. Python virtual
environment installation has been investigated as well to separate the CNTK
environment from other environments. Conda environments can be used as well as
Python virtual environments which will be explained in a separated post.
In any case, the CNTK installed from the source can be run much faster
than the pre-build binary packages provided by the Microsoft CNTK
 although it adds the complexity to installation process.
