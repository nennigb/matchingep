#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# matchingep : A python package for extending mode matching method on 2D
# waveguide tuned at an exceptional point.
# Copyright (C) 2021  B. Nennig (benoit.nennig@isae-supmeca.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Define the test suite based on doctest and unittest framework.


@author: bn
"""
import unittest
import doctest
import sys
import matchingep._matching

if __name__ == '__main__':
    # Define a test suite that can mix doctest and unittest
    # For now, only doctest
    print('> Running tests...')
    suite = doctest.DocTestSuite(matchingep._matching,
                                 optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE))
    # Define the runner
    runner = unittest.TextTestRunner(verbosity=3)
    # Run all the suite
    result = runner.run(suite)
    # Runner doesn't change exit status
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
