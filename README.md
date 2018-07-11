# fibergen

A FFT-based homogenization tool.

* FFT-based homogenization based on Lippmann-Schwinger equation with staggered grid approach \cite{SchneiderOspaldKabel2015:1}
* homogenization for linear elasticity, large deformations, Stokes flow and heat equation
* C++, OpenMP multiprocessing, XML + Python scripting interface
* laminate mixing for interfaces \cite{KabelMerkertSchneider2014,SchneiderOspaldKabel2015:2}
* mixed boundary conditions \cite{Kabel2016}
* generation of fibers distributions
* use of tetrahedrical mesh geometries
* arbitrarily many materials
* reading of raw CT data (LZ compressed)
* identification of homogenized material parameters
* ...


## Requirements

* gcc
* [boost](https://www.boost.org/) incl. boost.python 3 and [boost-numeric-bindings](https://mathema.tician.de/software/boost-numeric-bindings/)
* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/) incl. numpy headers
* [PyQT5](https://www.riverbankcomputing.com/software/pyqt/download5) incl. QWebEngine (QWebKiat also works)
* [lapack](www.netlib.org/lapack/) library


## Installation

1. download source
2. run build.sh, on error probably a library is missing
3. after successful build update your envirnoment variables:
```
export PATH=$PATH:$FIBERGEN/bin
export PYTHONPATH=$PYTHONPATH:$FIBERGEN/lib
```
where $FIBERGEN denotes your download directory.


## Run

```
fibergen-gui
```


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

