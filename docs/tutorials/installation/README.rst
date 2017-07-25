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
.. _Open MPI: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux#open-mpi

------------------------
Prepare the environment
------------------------

The following should be done in order:

    * CNTK system dependencies installation
    * CNTK Python dependencies installation
    * GPU prerequisites setup
    * `C++ Compiler`_
    * `Open MPI`_


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

The second line is for ``python3`` installation.

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

     export PATH=/usr/local/mpi/bin:$PATH export LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH


-----------------------
CNTK setup for Python
-----------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download the required binary package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _link: https://github.com/Microsoft/CNTK/releases

Please refer to this `link`_ for downloading desired binary packages.


~~~~~~~~~~~~~~~~~~~
pip installation
~~~~~~~~~~~~~~~~~~~

.. _link: https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python

Please refer to this `link`_ for different associated URLs for varieties of architecture.










~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Virtual Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, the environment must be activation. Since we already defined the environment alias as ``tensorflow``, by the terminal execution of the simple command of ``tensorflow``, the environment will be activated. Then like the previous part, we execute the following:

.. code:: bash

    pip install ~/tensorflow_package/file_name.whl

**WARNING**:
           * By using the virtual environment installation method, the sudo command should not be used anymore because if we use sudo, it points to native system packages and not the one available in the virtual environment.
           * Since ``sudo mkdir ~/virtualenvs`` is used for creating of the virtual environment, using the ``pip install`` returns ``permission error``. In this case, the root privilege of the environment directory must be changed using the ``sudo chmod -R 777 ~/virtualenvs`` command.

--------------------------
Validate the Installation
--------------------------

In the terminal, the following script must be run (``in the home directory``) correctly without any error and preferably any warning:

.. code:: bash

    python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))

--------------------------
Common Errors
--------------------------

Different errors reported blocking the compiling and running TensorFlow.

   * ``Mismatch between the supported kernel versions:`` This error mentioned earlier in this documentation. The naive solution reported being the reinstallation of the CUDA driver.
   * ``ImportError: cannot import name pywrap_tensorflow:`` This error usually occurs when the Python loads the tensorflow libraries from the wrong directory, i.e., not the version installed by the user in the root. The first step is to make sure we are in the system root such that the python libraries are utilized correctly. So basically we can open a new terminal and test TensorFlow installation again.
   * ``ImportError: No module named packaging.version":`` Most likely it might be related to the ``pip`` installation. Reinstalling that using ``python -m pip install -U pip`` or ``sudo python -m pip install -U pip`` may fix it!

--------------------------
Summary
--------------------------

In this tutorial, we described how to install TensorFlow from the source which has the advantage of more compatibility with the system configuration. Python virtual environment installation has been investigated as well to separate the TensorFlow environment from other environments. Conda environments can be used as well as Python virtual environments which will be explained in a separated post. In any case, the TensorFlow installed from the source can be run much faster than the pre-build binary packages provided by the TensorFlow although it adds the complexity to installation process.
