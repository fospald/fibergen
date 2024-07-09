<p align="center">
  <a href="LICENSE" alt="GPLv3 license"><img src="https://img.shields.io/badge/license-GPLv3-brightgreen.svg" /></a>
  <a href="#" alt="no warranty"><img src="https://img.shields.io/badge/warranty-no-red.svg" /></a>
  <a href="https://joss.theoj.org/papers/add4f02dcedfaae8f81da200000d64ee"><img src="https://joss.theoj.org/papers/add4f02dcedfaae8f81da200000d64ee/status.svg"></a>
  <a href="https://zenodo.org/badge/latestdoi/140197063"><img src="https://zenodo.org/badge/140197063.svg" alt="DOI"></a>
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


## Target Platform

The tool was developed and tested under Linux (Ubuntu). Other platforms such as Windows and MacOS might work but probably require adjustmest to the CMakeLists.txt file and some other small scripts.


## Requirements

The following libraries are required, which are likely already installed on your system:
* [CMake](https://cmake.org/)
* [gcc](https://gcc.gnu.org/) compiler (might also work with other compilers)
* [OpenMP](https://www.openmp.org/) for parallelization (optional)
* [boost](https://www.boost.org/) incl. boost.python 3 and [boost-numeric-bindings](https://mathema.tician.de/software/boost-numeric-bindings/)
* [FFTW3](http://www.fftw.org/) library
* [lapack](www.netlib.org/lapack/) library
* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/) incl. numpy headers
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) incl. QtWebEngine (QtWebKit also works)
* [zlib](https://zlib.net/) library
* [libpng](http://www.libpng.org/pub/png/libpng.html) library (optional for PNG output)

If unsure, continue with the installation and check the error messages of CMake.


## Docker Installation

1. Make sure you have [Docker and Docker compose](https://docs.docker.com/engine/install/ubuntu/) installed.
2. Build and run the container with GUI support in the background
```bash
cd docker
docker compose -f docker-compose-gui.yml up --detach
```
3. Run shell inside the container
```bash
sudo docker compose exec fibergen bash
```
4. Start fibergen as described in the [Run section](#run)


## Manual Installation

1. download source
```bash
git clone https://github.com/fospald/fibergen.git
```
2. run build.sh, on error probably a library is missing
```bash
sh build.sh [optional CMake parameters]
```
3. after successful build update your envirnoment variables:
```bash
export PATH=$PATH:$FIBERGEN/bin
export PYTHONPATH=$PYTHONPATH:$FIBERGEN/lib
```
where $FIBERGEN denotes your download directory.


## Run

Enter the following command to run the GUI (with an optional project file to load)
```bash
fibergen-gui [project.xml]
```
In order to run a project file from the command line run
```bash
fibergen project.xml
```
You can also run some test routines using
```bash
fibergen --test
```
in order to perform some internal tests of math and operators.


## Generating source code documentation

You can generate a [Doxygen](http://www.doxygen.org/)-based documentation by running 
```bash
cd doc/doxygen
make
firefox html/index.html
```


## Tutorial

Further information on how to use fibergen, can be found in [the tutorial](TUTORIAL.md) (also included in the doxygen documentation).


## Troubleshooting

### GUI Crash

There are known instances with QtWebKit which may result in a crash of the GUI.
An re-installation of QtWebKit with an older version or use of the newer QtWebEngine (i.e. using the latest version of Qt) may resolve the issue. Alternatively you can run the GUI with the demo- and help- browser disabled by
```bash
fibergen-gui --disable-browser
```
All QtWebKit/QtWebEngine browser instances will then be replaced by simplified QTextBrowser instances.


### Setting the Python version

If you get an error about "boost_python-pyXY" not found, try to figure out which Python version boost-python is compiled against by running
```bash
locate boost_python-py
```
and then modify the CMakeLists.txt accordingly
```bash
SET(PYTHON_VERSION_MAJOR X)
SET(PYTHON_VERSION_MINOR Y)
```

### Installing boost-numeric-bindings

Only the header files are required. No configure/build needed.
```bash
cd install_dir
git clone http://git.tiker.net/trees/boost-numeric-bindings.git
export BOOSTNUMERICBINDINGS_DIR=$(pwd)/boost-numeric-bindings
```
`install_dir` is the installation directory for boost-numeric-bindings. You should remove the `build` directory (`rm -r build`) before running `build.sh` again in order to clear the CMake cache.


## Contributing

If you have any question, idea or issue please create an new issue in the issue tracker.
If you want to contribute anything (e.g. demos) please contact me.


## Citing

You can use the following publication for citing fibergen:
```
@article{Ospald2019,
        author = {F. Ospald},
        title = {fibergen: An introductory tool for {FFT}-based material homogenization},
        year  = {2019},
        publisher = {The Open Journal},
        journal = {Journal of Open Source Software},
        volume = {4},
        number = {34},
        pages = {1027},
        doi = {10.21105/joss.01027}
}
```


## Acknowledgements

[Felix Ospald](https://www.tu-chemnitz.de/mathematik/part_dgl/people/ospald) gratefully acknowledges financial support by the [German Research Foundation](http://www.dfg.de/en/) (DFG), [Federal Cluster of Excellence EXC 1075](https://www.tu-chemnitz.de/MERGE/) "MERGE Technologies for Multifunctional Lightweight Structures". Many thanks to [Matti Schneider](https://www.itm.kit.edu/cm/287_3957.php) for his helpful introduction to FFT-based homogenization and ideas regarding the ACG distribution.

