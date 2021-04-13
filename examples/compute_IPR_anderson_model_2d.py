#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.1"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# compute_IPR_anderson_model_2d.py
# Authors: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 2d Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
# It computes the Inverse Participation Ratio of each state for a given realization of the disorder
# It then gathers 1/IPR vs. energy in the file "inverse_IPR.dat"
# And builds an histogram of the average value in "histogram_average_inverse_IPR.dat".
# The spectrum is strictyl bound by [-4-W/2,4+W/2]. The histrogram is build in this range.
# In the presence of disorder, the eigenstates are localized.
# The localization length is not known analytically, but huge for small W.
#
# -----------------------------------------------------------------------------------------
# if len(sys.argv) != 5:
#     print('Usage (4 parameters):\n compute_IPR_anderson_model_2d.py L W nr nsteps')
#     sys.exit()
# L = int(sys.argv[1])
# W = float(sys.argv[2])
# nr = int(sys.argv[3])
# nsteps = int(sys.argv[4])
L = 40
W = 3
nr = 10
nsteps = 50


# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L, W):
    disorder = W * ((np.random.uniform(size=L * L)).reshape((L, L)) - 0.5)
    return disorder


# Generate the Hamiltonian matrix for one realization of the random disorder
# The Hamitonian is in the LxL array "H"
def generate_hamiltonian(L, W):
    H = np.zeros((L * L, L * L))
    disorder = generate_disorder(L, W)
    for i in range(L):
        ip1 = (i + 1) % L
        for j in range(L):
            H[i * L + j, i * L + j] = disorder[i, j]
            jp1 = (j + 1) % L
            H[ip1 * L + j, i * L + j] = 1.0
            H[i * L + j, ip1 * L + j] = 1.0
            H[i * L + jp1, i * L + j] = 1.0
            H[i * L + j, i * L + jp1] = 1.0
    return H


def compute_IPR(eigenstates):
    IPR = np.sum(eigenstates ** 4, axis=0) * L ** 2 / np.sum(eigenstates ** 2, axis=0) ** 2
    return IPR


histogram = np.zeros(nsteps)
number_of_levels = np.zeros(nsteps, dtype=int)
energy_min = -4.0 - 0.5 * W
energy_step = (8.0 + W) / nsteps
IPR = np.zeros(nr * L * L)
energy_levels = np.zeros(nr * L * L)

for ir in range(nr):
    H = generate_hamiltonian(L, W)
    (energy_levels[ir * L * L:(ir + 1) * L * L], eigenstates) = linalg.eigh(H)
    IPR[ir * L * L:(ir + 1) * L * L] = compute_IPR(eigenstates)
# Sort energy_levels
idx = energy_levels.argsort()
energy_levels = energy_levels[idx]
IPR = IPR[idx]

filename = 'inverse_IPR.dat'
f = open(filename, 'w')
for i in range(nr * L * L):
    j = int((energy_levels[i] - energy_min) / energy_step)
    number_of_levels[j] += 1  # 统计能级在Emin~Emax的频次
    histogram[j] += 1.0 / IPR[i]
    f.write("%g %g\n" % (energy_levels[i], 1.0 / IPR[i]))
f.close()

print("Done, IPRs saved to", filename, "!")
filename = 'histogram_average_inverse_IPR.dat'
f = open(filename, 'w')
for j in range(nsteps):
    if (number_of_levels[j] > 0):
        histogram[j] /= number_of_levels[j]
    f.write("%g %g\n" % (energy_min + (j + 0.5) * energy_step, histogram[j]))
f.close()
print("Done, histogram of average IPR vs. energy saved to", filename, "!")

plt.figure(1)
m = np.linspace(0, nsteps-1, nsteps)
plt.scatter(m, histogram)
plt.show()
