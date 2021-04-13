#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
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
# diagonalization_anderson_model_2d.py
# Authors: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 2d Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
# Without disorder, the dispersion relation is energy=2*[cos(kx)+cos(ky)], with support on [-4,4],
# so that, for a  finite size system, the eigenstates are plane waves with
# kx=i*2*\pi/L, ky=j*2*\pi/L with i,j integers -L/2<i,j<=L/2.
# Eigenstates with +/i,+/-j are degenerate, allowing to build symmetric and antisymmetric combinations which
# are thus real wavefunctions.
# In the presence of disorder, the eigenstates are localized.
# The localization length is not known analytically, but huge for small W.
# The script computes and prints the full energy spectum for a single realization of the disorder
# It also prints the wavefunction of the state which has energy closest to the input parameter "energy"
# -----------------------------------------------------------------------------------------
if len(sys.argv) != 4:
  print('Usage (3 parameters):\n diagonalization_anderson_model_2d.py L W energy')
  sys.exit()
L = int(sys.argv[1])
W = float(sys.argv[2])
energy = float(sys.argv[3])

# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L,W):
  disorder=W*((np.random.uniform(size=L*L)).reshape((L,L))-0.5)
  return disorder

# Generate the Hamiltonian matrix for one realization of the random disorder
# The Hamitonian is in the LxL array "H"
def generate_hamiltonian(L,W):
  H=np.zeros((L*L,L*L))
  disorder=generate_disorder(L,W)
  for i in range(L):
    ip1=(i+1)%L
    for j in range(L):
      H[i*L+j,i*L+j]=disorder[i,j]
      jp1=(j+1)%L
      H[ip1*L+j  ,i  *L+j  ]=1.0
      H[i*  L+j  ,ip1*L+j  ]=1.0
      H[i  *L+jp1,i  *L+j  ]=1.0
      H[i*  L+j  ,i  *L+jp1]=1.0
  return H

# ____________________________________________________________________
# view_density.py
# Author: Dominique Delande
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# Reads 2d data in a ASCII file and make a color 2d plot with matplotlib
# In the ASCII file, the data must be on consecutive lines, following the rows
# of the 2d data, without hole
# The first two lines should contain "# n1 delta1" and "# n2 delta2" where
# n1 (n2) is the number of rows (columns)
# delta1 and delta2 are optional arguments given the step in the corresponding dimension
# Each line may contain several data, the last column is plotted,
# unless the there is a second argument, which is the column to be plotted
# ____________________________________________________________________

def view_density(file_name,column=-1,block=True):
  f = open(file_name,'r')
  #f = open('eigenstate.dat','r')
  line=(f.readline().lstrip('#')).split()
  n1=int(line[0])
  if len(line)>1:
    delta1=float(line[-1])
  else:
    delta1=1.0
  line=(f.readline().lstrip('#')).split()
  n2=int(line[0])
  if len(line)>1:
    delta2=float(line[-1])
  else:
    delta2=1.0
  #print sys.argv,len(sys.argv)
  arr=np.loadtxt(file_name,comments='#').reshape(n1,n2,-1)
  #print arr
  Z=arr[:,:,column]
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  plt.figure()
  plt.imshow(Z,origin='lower',interpolation='nearest')
  plt.show()
  return

# Generate a random Hamiltonian
H=generate_hamiltonian(L,W)
# Diagonalize it
(energy_levels,eigenstates)=linalg.eigh(H)
# The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
# Thus, sort them
idx = np.argsort(energy_levels)
energy_levels=energy_levels[idx]
eigenstates=eigenstates[:,idx]
# print the energy spectrum
filename= 'energy_spectrum.dat'
f=open(filename,'w')
for j in range(L*L):
  f.write("%g %g\n" % (j,energy_levels[j]))
f.close()
print("Done, energy spectrum saved to",filename,"!")
# Find the energy level closest to "energy"
i=np.argmin(abs(energy_levels-energy))
filename= 'density_eigenstate.dat'
f=open(filename,'w')
f.write("# %d\n" % L)
f.write("# %d\n" % L)
for j in range(L*L):
  f.write("%g %g\n" % (j,abs(eigenstates[j,i])**2))
f.close()
print("Done, eigenstate with energy",energy_levels[i],"saved to",filename,"!")
view_density('density_eigenstate.dat')

