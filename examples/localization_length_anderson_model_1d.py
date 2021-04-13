#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
__author__ = "Dominique Delande and Cord A. Mueller"
__copyright__ = "Copyright (C) 2017 Dominique Delande and Cord A. Mueller"
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
# localization_length_anderson_model_1d.py
# Authors: Dominique Delande and Cord A. Mueller
# Release date: March, 19, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 1D Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script computes the localization length as a function of energy.
# averaged over nr realizations.
# The equations to be solved are:  psi_{n+1} + psi_{n-1} + (w_n-E) psi_n = 0
# They are solved in the backward direction, starting from an outgoing wave of unit modulus,
# with normalization to unit incident flux at the end of the computation.
# The localization length is 12*(4-energy**2)/W**2 to lowest order in W.
# Typical values for numerical parameters are W=2.0, L=1000, nr=100, energy from -1.98 to +1.98 in 99 steps
# Increase to nr=1000 to obtain a less noisy curve
# -----------------------------------------------------------------------------------------
if len(sys.argv) != 7:
  print('Usage (6 parameters):\n localization_length_anderson_model_1d.py L W nr energy_min energy_max number_of_steps_in_energy')
  sys.exit()
L = int(sys.argv[1])
W = float(sys.argv[2])
nr = int(sys.argv[3])
energy_min = float(sys.argv[4])
energy_max = float(sys.argv[5])
nsteps=int(sys.argv[6])
filename= 'localization_length.dat'

def compute_psi(L,W,energy):
  k = math.acos(0.5*energy)
  exp_i_k = complex(math.cos(k),math.sin(k))
  psi=np.zeros(L+1,dtype=complex)
  psi[L] = exp_i_k  # 边界条件
  psi[L-1] = 1.0  # 边界条件
  for n in range(L-1,0,-1):
    w_n = W*(random.random()-0.5)
    psi[n-1] = (energy-w_n)*psi[n] - psi[n+1]
  return psi

def minuslogt(psi,energy):
  psi_minus_1 = energy*psi[0]-psi[1]
  k = math.acos(0.5*energy)
  exp_i_k = complex(math.cos(k),math.sin(k))
  incident = (0.5*abs(psi_minus_1-exp_i_k*psi[0])/math.sin(k))**2
  return math.log(incident)

def compute_transmission(L,W,energy,nr):
  minus_log_transmission=np.zeros(nr)
  for i in range(nr):
    psi=compute_psi(L,W,energy)
    minus_log_transmission[i]=minuslogt(psi,energy)
  return minus_log_transmission

f=open(filename,'w')

energy_step=(energy_max-energy_min)/nsteps
for i in range(nsteps+1):
  energy=energy_min+i*energy_step
  minus_log_transmission=compute_transmission(L,W,energy,nr)
  localization_length=L/(np.sum(minus_log_transmission)/nr)
  f.write("%g %g\n" % (energy,localization_length))
  print("Done energy = ",energy)
f.close()

print("Done, result saved to",filename,"!")