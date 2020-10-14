CMake C++ Installation
======================

We recommend to use `CMake` for the installation since it is the most automated way to reach your needs.
First of all make sure you have a sufficient version of `CMake` installed (3.9 minimum version required).
If you are working on a machine without root privileges and you need to upgrade your `CMake` version a valid solution to overcome your problems is provided shut_.

With a valid `CMake` version installed first of all clone the project as:

.. code-block:: bash

	git clone https://github.com/Nico-Curti/DNetPRO
	cd DNetPRO


The you can build the `DNetPRO` package with

.. code-block:: bash

	mkdir -p build
	cd build && cmake .. && cmake --build . --target install

or more easily

.. code-block:: bash

	./build.sh

if you are working on a Windows machine the right script to call is the `build.ps1`_.

.. note::
  If you want enable the OpenMP support (*4.5 version is required*) compile the library with `-DOMP=ON`.

.. note::
	If you want enable the Cython support compile the library with `-DPYWRAP=ON`. The Cython packages will be compiled and correctly positioned in the `DNetPRO` Python package.

.. _shut: https://github.com/Nico-Curti/Shut
.. _`build.ps1`: https://Nico-Curti/DNetPRO/blob/master/build.ps1
