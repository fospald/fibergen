---
title: '``fibergen``: An introductory tool for FFT-based material homogenization'
tags:
  - material homogenization
  - FFT
  - Python
  - C++
  - heat equation
  - Stokes flow
  - Darcy flow
  - linear elasticity
  - hyperelasticity
authors:
  - name: Felix Ospald
    orcid: 0000-0001-8372-9179
    affiliation: "1"
affiliations:
 - name: Research Group Numerical Mathematics (Partial Differential Equations), TU Chemnitz, Chemnitz, Germany
   index: 1
date: 2 October 2018
bibliography: paper.bib
---

# Summary

Many engineering applications require material properties of heterogeneous media. For instance elasticity constants and thermal conductivity of fiber reinforced plastics, the viscosity of a suspension, permeability of porous media and the electrical conductivity of sintered metals.
Laboratory experiments for determining these properties are often difficult, expensive and error-prone.
Therefore, increasingly computer simulations are used instead, both for the computation of effective properties and the generation of the geometry of such materials in order to avoid the use of expensive computer tomography (CT) scans.

[``fibergen``](http://fospald.github.io/fibergen/) is a easy to use tool (command-line/GUI) for the homogenization (i.e. obtaining effective material parameters of an equivalent homogeneous media) and generation of periodic materials, especially but not limited to fiber reinforced composites produced by injection molding.
The voxel-based spectral method relies on the fast Fourier transform (FFT) and OpenMP parallelization for highly efficient computations, which can be carried out on an average desktop computer. The tool also includes a geometry generator, but is also capable to process raw data (e.g. CT data).

# Relation to similar projects

There exists a number of similar projects for the homogenization of materials, developed by well-known people in this research area:

* [AMITEX](http://www.maisondelasimulation.fr/projects/amitex/html), Lionel Gélébart, 
Atomic Energy and Alternative Energies Commission, France, (Fortran)
* [morphhom](http://cmm.ensmp.fr/morphhom), François Willot, Université Paris, France, (Fortran/C)
* [DAMASK](http://damask.mpie.de/Documentation/SpectralSolver), Martin Diehl, Max-Planck-Institut für Eisenforschung, Germany, (Fortran, C, C++)
* [janus](http://github.com/sbrisard/janus), Sébastien Brisard, Université Paris, France, (Cython)
* [FFTHomPy](http://github.com/vondrejc/FFTHomPy), Jaroslav Vondřejc, TU Braunschweig, Germany (Python)

All these projects are written in different programming languages and implement different solver methods and provide different features. Most of these tools provide only the pure solver without a geometry generator and without post-processing (visualization) of the results.
``fibergen`` on the other hand provides a solver but also an GUI/IDE for editing project files, running the solver and post-processing.
It is also written in object-oriented C++ and provides a simple Python interface for the automation of computations.

# Features

``fibergen`` implements FFT-based homogenization based on the Lippmann-Schwinger equation.
It includes the original method by [@MoulinecSuquet1994] and the more recent finite difference discretizations on a rotated grid by [@Willot2015] and on a staggered grid by [@SchneiderOspaldKabel2015]. Especially the staggered grid method is not implemented by any other of the listed tools, although it is the superior method [@SchneiderOspaldKabel2015]. ``fibergen`` also implements efficient fixed point and Newton-Krylov solvers for FFT-based homogenization of elasticity at large deformations, as described in [@KabelBöhlkeSchneider2014].
Further it implements so called composite voxels, for the linear [@KabelMerkertSchneider2015] and nonlinear case [@SchneiderOspaldKabel2015], which increases the accuracy and/or speed of the method enormously.
Also mixed boundary conditions [@Kabel2016] are supported.
Further homogenization solvers for the effective thermal conductivity, porosity and viscosity are provided.
Geometry can be loaded from (gzip compressed) raw data (e.g. CT scans) or also be generated from simple sphere, cylinder and capsule-like objects, but also tetrahedral meshes can be used as input geometry.
For cylinder and capsule objects a orientation distribution can be selected for the generation, as well as distributions for the spatial position, length- and diameter.
Here we provide the angular central Gaussian distribution as employed by [@Montgomery-SmithJackSmith2011] amongst other.

Projects are stored as XML configuration files, which may be executed over the command-line or using the GUI.
Furthermore ``fibergen`` provides a Python library, which can be used to create solver objects, modify settings, run the solver and extract results in an automated fashion.

![``fibergen`` GUI, showing the main screen for selecting one of the predefined demos.](../page/images/screenshot_1.png)

![``fibergen`` GUI, showing the project editor on the left and the postprocessing view on the right.](../page/images/screenshot_2.png)

# Development

The ``fibergen`` core was developed in C++, heavily using the [Boost](http://www.boost.org) and [boost-numeric-bindings](http://mathema.tician.de/software/boost-numeric-bindings/) for Boost uBLAS libraries.
Parallelization is realized through [OpenMP](http://www.openmp.org/).
The Python 2/3 compatible interface is provided using [Boost.Python](http://www.boost.org/doc/libs/release/libs/python/), where computed results are returned as [numpy](http://www.numpy.org/)-array.
The graphical user interface was developed using [PyQt5](http://pypi.org/project/PyQt5/), the Python bindings of [Qt5](http://doc.qt.io/qt-5/qt5-intro.html), including the [QWebEngine](http://doc.qt.io/qt-5.11/qtwebengine-index.html) (or QWebKit) for browsing demos as well as the Qt back-end of [matplotlib](http://matplotlib.org/) for embedding plots. A small amount of [scipy](http://www.scipy.org/) functionality is used to write images.

# Limitations

``fibergen`` does not implement advanced methods for crystal-plasticity or even cracking.
It should be seen as an simple tool for FFT-based homogenization, with a simple-to-use GUI
where different methods and parameters can be selected and compared easily.
The geometry generation tool is definitely limited to simple arrangements of objects.
The nonlinear solvers also have some limitations/convergence issues for certain problems.
Nevertheless it can solve problems with several 100 million unknowns easily, provided enough memory is available.
The computed results are without any liability and warranty for their correctness.

# Acknowledgements

[Felix Ospald](http://www.tu-chemnitz.de/mathematik/part_dgl/people/ospald) gratefully acknowledges financial support by the [German Research Foundation](http://www.dfg.de/en/) (DFG), [Federal Cluster of Excellence EXC 1075](http://www.tu-chemnitz.de/MERGE/) "MERGE Technologies for Multifunctional Lightweight Structures". Many thanks to [Matti Schneider](http://www.itm.kit.edu/cm/287_3957.php) for his helpful introduction to FFT-based homogenization and ideas regarding the ACG distribution.

# References

