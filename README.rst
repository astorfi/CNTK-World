
***************
CNTK World
***************
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/Keras-Examples/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=102
    :target: https://github.com/ellerbrock/open-source-badge/


This repository is aimed to provide simple and ready-to-use tutorials for CNTK. Each tutorial has a ``source code``.

.. image:: _img/mainpage/CNTK-World.gif

.. The links.
.. .. _wiki: https://github.com/astorfi/TensorFlow-World/wiki

#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 4

============
Motivation
============

~~~~~~~~~~~~~~~~~
Why using CNTK?
~~~~~~~~~~~~~~~~~
Deep Learning is of great interest these days - there's a need for rapid and optimized implementations
of the algorithms and deep architectures. `Microsoft Cognitive Toolkit (CNTK)`_ is designed to provide a free
and fast-and-easy platform for facilitating the deep learning architecture design and implementation.
CNTK demonstrated to be superior compared to the famous TensorFlow in performance (**Benchmarking State-of-the-Art Deep Learning Software Tools**: `report`_, `paper`_).

.. Benchmarking State-of-the-Art Deep Learning Software Tools
.. _report: http://dlbench.comp.hkbu.edu.hk/
.. _paper: https://arxiv.org/pdf/1608.07249.pdf
.. _Microsoft Cognitive Toolkit (CNTK): https://docs.microsoft.com/en-us/cognitive-toolkit/reasons-to-switch-from-tensorflow-to-cntk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What's the point of this repository?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CNTK`` is new and it's hard to find easy-to-use code examples and tutorials for *learning by doing*!
The Microsoft itself provided a nice comprehensive `Tutorials Series`_ on what is CNTK and how to design
and implement the deep architectures. However, sometimes its tutorials may become too verbose and complicated especially
with *data-reader* objects and preprocessing phases. So there is a need for an open-source project to satisfy the followings:

  1. Covers the basic models.
  2. Be as simple as possible but not simpler than what is required!
  3. Be actively underdeveloped by the people of GitHub and not only the people on Microsoft!
  4. Examples must be run with one push of a button and not more!
  5. And etc, which needs to be added by people who are following this project!

We hope that those aforementioned lines would be satisfied in this project.

.. _Tutorials Series: https://cntk.ai/pythondocs/tutorials.html


================
CNTK Tutorials
================
The tutorials in this repository are partitioned into relevant categories.

==========================

~~~~~~~~~~~~~~~~~~~~~
PART 0 - Installation
~~~~~~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/installation.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right
   :target: https://github.com/astorfi/CNTK-World/tree/master/docs/tutorials/installation


+--------------------------------------+-------------------------------------------------+
| *Installation*                       | `CNTK Installation`_                            |
+--------------------------------------+-------------------------------------------------+

==========================

~~~~~~~~~~~~~~~
PART 1 - Basics
~~~~~~~~~~~~~~~

.. image:: _img/mainpage/basics.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 1  | *Start-up*                        | `Python <welcomesourcecode_>`_     / `IPython <ipythonwelcome_>`_                             |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 2  | *Basic Operations*                | `Python <basicoperationpython_>`_  / `IPython <ipythonbasicoperation_>`_                      |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+

==========================

~~~~~~~~~~~~~~~~~~~~~
PART 2 - Basic Models
~~~~~~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/logisticregression.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 1  | *Linear Regression*               | `Python <linearregressionpython_>`_     / `IPython <ipythonlinearregression_>`_               |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 2  | *Logistic Regression*             | `Python <logisticregressionpython_>`_   / `IPython <ipythonlogisticregression_>`_             |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+

==========================

~~~~~~~~~~~~~~~~~~~~~~~~~
PART 3 - Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/CNNs.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 1  | *Multi Layer Perceptron*          | `Python <MLPpython_>`_                              / `IPython <ipythonMLP_>`_                |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 2  | *Convolutional Neural Networks*   | `Python <CNNpython_>`_                              / `IPython <ipythonCNN_>`_                |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 3  | *Autoencoders (undercomplete)*    | `Python <AEUpython_>`_                              / `IPython <ipythonAEU_>`_                |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+


==========================

~~~~~~~~~~~~~~~~~~~~~~~~~
PART 4 - Advanced
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/advanced.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+-----------------------------------+-----------------------------------------------------------------------------------------------+
| 1  | *Conditional GAN*                 | `Python <CGANpython_>`_                             / `IPython <ipythonCGAN_>`_               |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+

==========================


.. ~~~~~~~~~~~~
.. **Welcome**
.. ~~~~~~~~~~~~

.. The tutorial in this section is just a simple entrance to TensorFlow world.

.. _welcomesourcecode: codes/Python/p01-warmup/0-welcome/welcome.py
.. _ipythonwelcome: codes/IPython/p01-warmup/0-welcome/welcome.ipynb

.. _basicoperationpython: codes/Python/p01-warmup/1-basicoperations/basicoperations.py
.. _ipythonbasicoperation: codes/IPython/p01-warmup/1-basicoperations/basicoperations.ipynb

.. ~~~~~~~~~~~~
.. **Basic Models**
.. ~~~~~~~~~~~~

.. _linearregressionpython: codes/Python/p02-basic-models/1-linear-regression/linear-regression.py
.. _ipythonlinearregression: codes/IPython/p02-basic-models/1-linear-regression/linear-regression.ipynb

.. _logisticregressionpython: codes/Python/p02-basic-models/2-logistic-regression/logistic-regression.py
.. _ipythonlogisticregression: codes/IPython/p02-basic-models/2-logistic-regression/logistic-regression.ipynb

.. ~~~~~~~~~~~~
.. **Neural**
.. ~~~~~~~~~~~~

.. _MLPpython: codes/Python/p03-neural-networks/1-multilayer-perceptron/multilayer-perceptron.py
.. _ipythonMLP: codes/IPython/p03-neural-networks/1-multilayer-perceptron/multilayer-perceptron.ipynb

.. _CNNpython: codes/Python/p03-neural-networks/2-convolutional-neural-networks/convolutional-nn.py
.. _ipythonCNN: codes/IPython/p03-neural-networks/2-convolutional-neural-networks/convolutional-nn.ipynb

.. _AEUpython: codes/Python/p03-neural-networks/3-autoencoders/autoencoders.py
.. _ipythonAEU: codes/IPython/p03-neural-networks/3-autoencoders/autoencoders.ipynb


.. ~~~~~~~~~~~~
.. **Advanced**
.. ~~~~~~~~~~~~

.. _CGANpython: codes/Python/p04-advanced/1-conditional-DCGAN/conditional-DCGAN.py
.. _ipythonCGAN: codes/IPython/p04-advanced/1-conditional-DCGAN/conditional-DCGAN.ipynb



=============================================
CNTK Installation and Setup the Environment
=============================================

.. _CNTK Installation: docs/tutorials/installation

In order to install CNTK please refer to the following link:

  * `CNTK Installation`_


.. .. image:: _img/mainpage/installation.gif
    :target: https://www.youtube.com/watch?v=_3JFEPk4qQY&t=2s


The virtual environment installation is recommended in order to prevent package
conflict and having the capacity to customize the working environment. Among different
methods of creating and utilizing virtual environments, working with ``conda`` is
recommended.

.. =====================
.. Some Useful Tutorials
.. =====================

  .. * `TensorFlow Examples <https://github.com/aymericdamien/TensorFlow-Examples>`_ - TensorFlow tutorials and code examples for beginners
  .. * `Sungjoon's TensorFlow-101 <https://github.com/sjchoi86/Tensorflow-101>`_ - TensorFlow tutorials written in Python with Jupyter Notebook
  .. * `Terry Um’s TensorFlow Exercises <https://github.com/terryum/TensorFlow_Exercises>`_ - Re-create the codes from other TensorFlow examples
  .. * `Classification on time series <https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition>`_ - Recurrent Neural Network classification in TensorFlow with LSTM on cellphone sensor data



=============
Contributing
=============

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. *For typos, please
do not create a pull request. Instead, declare them in issues or email the repository owner*.

Please note we have a code of conduct, please follow it in all your interactions with the project.

~~~~~~~~~~~~~~~~~~~~
Pull Request Process
~~~~~~~~~~~~~~~~~~~~

Please consider the following criterions in order to help us in a better way:

  * The pull request is mainly expected to be a code script suggestion or improvement.
  * A pull request related to non-code-script sections is expected to make a significant difference in the documentation. Otherwise, it is expected to be announced in the issues section.
  * Ensure any install or build dependencies are removed before the end of the layer when doing a build and creating a pull request.
  * Add comments with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
  * You may merge the Pull Request in once you have the sign-off of at least one other developer, or if you do not have permission to do that, you may request the owner to merge it for you if you believe all checks are passed.

~~~~~~~~~~~
Final Note
~~~~~~~~~~~

We are looking forward to your kind feedback. **Please help us to improve this open source project and make our work better.
For contribution, please create a pull request and we will investigate it promptly**. Once again, we appreciate
your kind feedback and elaborate code inspections.
