MatchingEP
==========

[![tests](https://github.com/nennigb/matchingep/actions/workflows/ci-ubuntu.yml/badge.svg)](https://github.com/nennigb/matchingep/actions/workflows/ci-ubuntu.yml)

**A python package for extending mode matching method on a 2D waveguide tuned at an exceptional point.**

This package aims at solving the wave propagation in a 2D (acoustic) waveguide comprising two
semi-infinite sections using the enhanced _mode matching method_ method proposed in [^1]. One duct has rigid boundaries whilst the other is lined along both walls (Robin boundary condition). 

![Lined duct sketch.](doc/lined_duct.svg)

The **mode-matching method** is a convenient tool to compute multimodal scattering coefficients between
connected waveguides. In each waveguides, the field is expanded on a finite set of propagating and evanescent waves and the solution of the global problem is obtained by solving a linear system deduced from the continuity conditions at the interfaces. The propagating and evanescent waves are obtained by solving the dispersion equation.

The method is well established when all roots of the dispersion equation are simple but fails when two or more eigenvectors merge. **Exceptional points** (EP) correspond to particular values of the parameters (here admittances) leading to defective eigenvalue in non-Hermitian systems. At EP, both the eigenvalues and eigenvectors merge.

The proposed approach extends the mode-matching method at **exceptional point** (EP) by adding new wavefunctions, with polynomial growth along the guide axis as proposed in [^1]. With these functions, the basis is complete and pointwise convergence is recovered. These function are analogous to Jordan generalized vectors in a finite-dimensional vector space.

This package works for EP2 (two modes merging) or EP3 (three modes merging).

[^1]: J. B Lawrie, B. Nennig, E. Perrey-Debain, 2022, [10.1098/rspa.2022.0484](https://doi.org/10.1098/rspa.2022.0484).

The `examples` folder contains most of scripts to obtained results presented in [^1].
The package proposes some facilties to compute the power balance, solving the dispersion equation, plotting...


## Installation

First, install the [`polze`](https://github.com/nennigb/polze/) package (root finding). Then, installation of `matchingep` can be done after cloning or downloading the repos, using 
```
pip3 install path/to/matchingep [--user]
```
or in _editable_ mode if you want to modify the sources
```
pip3 install -e path/to/matchingep
```
Note that on some systems (like ubuntu), `python3` or `pip3` should be use to call python v3.x


## Running test suite

The test suite based on doctest and unittest can be run, with
```
python -m matchingep.test
```

## Documentation

The doctrings are compatible with several Auto-generate API documentation, like `pdoc3`. Once the package has been installed or the at `matchingep` location run,
```
pdoc3 --html --force --config latex_math=True matchingep
```
The html files are generated in place. Then open the `matchingep/index.html` file. This interactive doc is particularly useful to see latex includes.


## Usage

For a lined duct with unit height, lined with the two admittances `mu` and `nu`, the problem can be solved 
```python
import numpy as np
import matchingep as matching 
# Define the admittances (here EP3 case)
mu3 = 3.08753629148967+3.62341792246081j
nu3 = 3.17816250726595+4.67518038763374j
# Define the number of modes in the expansion
Nmodes = 30
# Define the radius for the countour solver (dispersion equation)
Radius = (min(10, Nmodes) + 1.2)*np.pi
# Create the `guide` object
guide = matching.Guide(mu=mu3,
                       nu=nu3, k=1,
                       Nmodes=Nmodes, R=Radius)
# Chose a matching scheme depending of the root multiplicities
EP = np.rint(guide.multiplicities).max()
guide.matching(EP_order=EP)
# Get the transmitted (x=1) and reflected power
T = guide.transmitted_power(x=1)
R = guide.reflected_power()
```

## Citing
If you are using `matchingep` in your scientific research, please cite

> J. B. Lawrie, B. Nennig, and E. Perrey-Debain. "Analytic mode-matching for accurate handling of exceptional points in a lined acoustic waveguide." Proceedings of the Royal Society A 478.2268 (2022): 20220484, [10.1098/rspa.2022.0484](https://doi.org/10.1098/rspa.2022.0484).

BibTex:
```bibtex
@article{doi:10.1098/rspa.2022.0484,
  author = {Lawrie, Jane B. and Nennig, B. and Perrey-Debain, E.},
  title = {Analytic mode-matching for accurate handling of exceptional points in a lined acoustic waveguide},
  journal = {Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences},
  volume = {478},
  number = {2268},
  pages = {20220484},
  year = {2022},
  doi = {10.1098/rspa.2022.0484}
}
```


## License

`matchingep` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
`matchingep` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with `matchingep`.  If not, see <https://www.gnu.org/licenses/>.
