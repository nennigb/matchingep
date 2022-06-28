#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    matchingep : A python package for extending mode matching method on 2D
#    waveguide tuned at an exceptional point.
#    Copyright (C) 2021  B. Nennig (benoit.nennig@isae-supmeca.fr)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='matchingep',
      version='1.0',
      author_email="benoit.nennig@isae-supmeca.fr",
      description="A python package for extending mode matching method on a 2D waveguide tuned at an exceptional point.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'polze',
                        'tqdm'],  # Use in examples
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL 3",
        "Operating System :: OS Independent"],
      python_requires='>3.6')
