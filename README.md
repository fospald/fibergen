<p align="center">
  <a href="LICENSE" alt="GPLv3 license"><img src="https://img.shields.io/badge/license-GPLv3-brightgreen.svg" /></a>
  <a href="#" alt="no warranty"><img src="https://img.shields.io/badge/warranty-no-red.svg" /></a>
</p>

# fibergen

A FFT-based homogenization tool.

* FFT-based homogenization based on Lippmann-Schwinger equation with staggered grid approach ([SchneiderOspaldKabel2015:1](http://dx.doi.org/10.1002/nme.5008))
* homogenization for linear elasticity, large deformations, Stokes flow and heat equation
* C++, OpenMP multiprocessing, XML + Python scripting interface
* laminate mixing for interfaces ([KabelMerkertSchneider2014](http://dx.doi.org/10.1016/j.cma.2015.06.003), [SchneiderOspaldKabel2015:2](http://dx.doi.org/10.1016/j.cma.2016.06.021))
* mixed boundary conditions ([Kabel2016](http://dx.doi.org/10.1007/s00466-015-1227-1))
* generation of fibers distributions
* use of tetrahedrical mesh geometries
* arbitrarily many materials
* reading of raw CT data (gzip compressed)
* identification of homogenized material parameters
* ...


## Requirements

The following libraries are required, which are likely already installed on your system:
* [CMake](https://cmake.org/)
* [gcc](https://gcc.gnu.org/)
* [boost](https://www.boost.org/) incl. boost.python 3 and [boost-numeric-bindings](https://mathema.tician.de/software/boost-numeric-bindings/)
* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/) incl. numpy headers
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) incl. QWebEngine (QWebKit also works)
* [lapack](www.netlib.org/lapack/) library

If unsure, continue with the installation and check the error messages of CMake.


## Installation

1. download source
```
git clone https://github.com/fospald/fibergen.git
```
2. run build.sh, on error probably a library is missing
```
sh build.sh [optional CMake parameters]
```
3. after successful build update your envirnoment variables:
```
export PATH=$PATH:$FIBERGEN/bin
export PYTHONPATH=$PYTHONPATH:$FIBERGEN/lib
```
where $FIBERGEN denotes your download directory.


## Run

Enter the following command to run the GUI (with an optional project file to load)
```
fibergen-gui [project.xml]
```
In order to run a project file from the command line run
```
fibergen project.xml
```
You can also run some test routines using
```
fibergen --test
```
in order to perform some internal tests of math and operators.


## Troubleshooting

### Setting the Python version

If you get an error about "boost_python-pyXY" not found, try to figure out which Python version boost-python is compiled against byrunning
```
locate boost_python-py
```
and then modify the CMakeLists.txt accordingly
```
SET(PYTHON_VERSION_MAJOR X)
SET(PYTHON_VERSION_MINOR Y)
```

### Installing boost-numeric-bindings

Only the header files are required. No configure/build needed.
```
cd install_dir
git clone http://git.tiker.net/trees/boost-numeric-bindings.git
export BOOSTNUMERICBINDINGS_DIR=$(pwd)/boost-numeric-bindings
```

## Contributing

If you have any question, idea or issue please create an new issue in the issue tracker.
If you want to contribute anything (e.g. demos) please contact me.


## Acknowledgements

[Felix Ospald](https://www.tu-chemnitz.de/mathematik/part_dgl/people/ospald) gratefully acknowledges financial support by the [German Research Foundation](http://www.dfg.de/en/) (DFG), [Federal Cluster of Excellence EXC 1075](https://www.tu-chemnitz.de/MERGE/) "MERGE Technologies for Multifunctional Lightweight Structures". Many thanks to [Matti Schneider](https://www.itm.kit.edu/cm/287_3957.php) for his helpful introduction to FFT-based homogenization and ideas regarding the ACG distribution.

