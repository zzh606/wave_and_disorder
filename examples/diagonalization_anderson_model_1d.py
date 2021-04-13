#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from scipy import linalg
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
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
# diagonalization_anderson_model_1d.py
# Authors: Dominique Delande
# Release date: March, 19, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 1d Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
# Without disorder, the dispersion relation is energy=2*cos(k), with support on [-2,2],
# so that, for a  finite size system, the eigenstates are plane waves with
# k=i*2*\pi/L, with i an integer -L/2<i<=L/2.
# Eigenstates with +/i are degenerate, allowing to build symmetric and antisymmetric combinations which
# are thus real wavefunctions.
# In the presence of disorder, the eigenstates are localized.
# The localization length is 12*(4-energy**2)/W**2 to lowest order in W.
# The script computes and prints the full energy spectum for a single realization of the disorder
# It also prints the wavefunction of the state which has energy closest to the input parameter "energy"
# -----------------------------------------------------------------------------------------
if len(sys.argv) != 4:
  print('Usage (3 parameters):\n diagonalization_anderson_model_1d.py L W energy')
  sys.exit()
L = int(sys.argv[1])
W = float(sys.argv[2])
energy = float(sys.argv[3])

# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L,W):
  disorder=W*(np.random.uniform(size=L)-0.5)
  return disorder

# Generate the Hamiltonian matrix for one realization of the random disorder
# The Hamitonian is in the LxL array "H"
def generate_hamiltonian(L,W):
  H=np.zeros((L,L))
  disorder=generate_disorder(L,W)
  for i in range(L):
# the following line to ensure periodic boundary conditions
    ip1=(i+1) %L
    H[i,i]=disorder[i]
    H[i,ip1]=1.0
    H[ip1,i]=1.0
  return H

# Generate a random Hamiltonian
H=generate_hamiltonian(L,W)
# Diagonalize it
(energy_levels,eigenstates)=linalg.eigh(H)
# The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
# Thus, sirt them
idx = energy_levels.argsort()[::1]
energy_levels=energy_levels[idx]
eigenstates=eigenstates[:,idx]
# print the energy spectrum
filename= 'energy_spectrum.dat'
f=open(filename,'w')
for j in range(L):
  f.write("%g %g\n" % (j,energy_levels[j]))
f.close()
print("Done, energy spectrum saved to",filename,"!")
# Find the energy level closest to "energy"
i=np.argmin(abs(energy_levels-energy))
filename= 'eigenstate.dat'
f=open(filename,'w')
for j in range(L):
  f.write("%g %g\n" % (j,eigenstates[j,i]))
f.close()
print("Done, eigenstate with energy",energy_levels[i],"saved to",filename,"!")
