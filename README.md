# fibergen

A FFT-bases homogenization tool.

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
* boost incl. boost.python 3
* Python 3
* scipy
* PyQT incl. WebEngine

## Installation

* 1. download source
* 2. run build.sh, on errors probably a library is missing
* 3. after successful build update your envirnoment variables:
```
export PATH=$PATH:$FIBERGEN/bin
export PYTHONPATH=$PYTHONPATH:$FIBERGEN/lib
```
where $FIBERGEN denotes your download directory.

## Run

```
fibergen-gui
```

