#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from matplotlib import pyplot as plt
import time

__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2017"
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
# classical_diffusion_2D.py
#
# Author: Dominique Delande
# Release date: Mar, 3, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# ----------------------------------------------------------------
# This script models a classical multiple scattering evolution in a 2D system
# The scattering mean free path is chosen equal to the unit of length
# ----------------------------------------------------------------
# This script can be ran with python 2 or 3. It requires numpy.
# It simulates the classical propagation of particles in a 2D random environment.
# Each trajectory is composed of straight segments interleaved with scattering
# events. The length of each segment is chosen following an exponential
# distribution, with a unit mean free path. The scattering angle is chosen
# with a Gaussian distribution around 0, with a sigma equal to the
# "typical_scattering_angle" parameter. The initial point is at the origin,
# the initial direction of propagation random.
# No seed in the random number generation, but this can be easily added (line is
# commented out).
# The script propagates "number_of_paths" trajectories during a time
# "propagation_time" (which is actually a number of scattering events).
#
# The script outputs 4 files:
# * trajectories.dat which contains the (x,y) coordinates of the first 10
# trajectories.
# * squared_displacement.dat, which contains <x^2+y^2> (averaged over all
# trajectories) vs. time
# * final_position_x_distribution.dat and final_position_y_distribution.dat which
# contain the distribution of final positions. This is an histogram covering
# the range [-"system_size"/2,-"system_size"/2]. There are 100 bins, so this makes
# sense only if there are much more trajectories.
#
# The theoretical predictions for the squared_displacement and the final_position
# distributions are the following:
# * <r^2(t)> = 2Dt with D=1/(1-exp(-0.5*typical_scattering_angle^2)). D=1 in the
# isotropic limit, 2/typical_scattering_angle^2 for very anistropic scattering.
# * P(x) and P(y) are Gaussians with sigma=sqrt(Dt).
# -----------------------------------------------------------------------
# Read parameters from the input line:
if len(sys.argv) != 5:
    print(
        'Usage (4 parameters): classical_diffusion_2D.py typical_scattering_angle propagation_time number_of_paths system_size')
    print('       typical_scattering_angle governs the anistropy of the scattering events')
    print('          if much larger than 2*pi, isotropic scattering')
    print('          if small, scattering is mainly in the forward direction')
    print('       Typical values for numerical parameters are')
    print('       propagation_time=100, number_of_paths=10, system_size=100, to see few multiple scattering paths')
    print('       propagation_time=100, number_of_paths=10000, system_size=100, to compute an histogram')
    print('                    of the final position distribution or compute the diffusion coefficient')
    sys.exit()
# Typical values for numerical parameters are
# propagation_time=100, number_of_paths=10 to see few multiple scattering paths
# propagation_time=100, number_of_paths=10000 to compute an histogram of the final position distribution or compute the diffusion coefficient
typical_scattering_angle = float(sys.argv[1])
propagation_time = int(sys.argv[2])
number_of_paths = int(sys.argv[3])
system_size = float(sys.argv[4])
number_of_bins = 100
number_of_printed_trajectories = 10
# The random number generator can be made reproducible by setting the seed
# np.random.seed(42)

position_x = np.zeros(propagation_time + 1)
position_y = np.zeros(propagation_time + 1)
theta = np.zeros(propagation_time)
path_length = np.zeros(propagation_time)
histogram_x = np.zeros(number_of_bins)
histogram_y = np.zeros(number_of_bins)
r2 = np.zeros(propagation_time + 1)

# Cheat code
# If system_size is negative, compute in addition the analytic results in some other files
if (system_size < 0.0):
    system_size = -system_size
    cheat = True
else:
    cheat = False

histogram_bin = system_size / number_of_bins
half_system_size = 0.5 * system_size


# This computes a single multiple scattering path starting from position (0,0) for propagation_time steps
def compute_path():
    position_x[0] = 0
    position_y[0] = 0
    # initial angle is randomized
    current_theta = np.random.rand() * 2.0 * math.pi
    # random angle
    theta = np.random.normal(scale=typical_scattering_angle, size=propagation_time)
    # length of each step
    path_length = np.random.exponential(size=propagation_time)
    for i in range(propagation_time):
        current_theta += theta[i]
        position_x[i + 1] = position_x[i] + path_length[i] * math.cos(current_theta)
        position_y[i + 1] = position_y[i] + path_length[i] * math.sin(current_theta)
    return (position_x, position_y)


# my code
tra_x = []
tra_y = []

position_x, position_y = compute_path()

plt.figure(1)
# for i in range(len(position_x)):
#     plt.plot(position_x[i], position_y[i])
#     plt.show()
#     time.sleep(0.01)
for i in range(number_of_paths):
    position_x, position_y = compute_path()
    plt.plot(position_x, position_y)
plt.show()
# time.sleep(100)

position_file = 'trajectories.dat'
f = open(position_file, 'w')
f.write('# Multiple scattering paths in the plane\n')
f.write('# There are %d different trajectories, each with the (x,y) coordinates of the scattering events\n' % (
    number_of_printed_trajectories))
f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
f.write('# Length of each trajectory = %d\n' % (propagation_time))
for j in range(number_of_paths):
    position_x, position_y = compute_path()
    r2 = r2 + position_x * position_x + position_y * position_y
    # output only the first 10 trajectories
    if (j < number_of_printed_trajectories):
        for i in range(propagation_time + 1):
            f.write('%12.8g %12.8g\n' % (position_x[i], position_y[i]))
        f.write('\n\n')
    # histogram of final x positions
    ix = math.floor((position_x[propagation_time] + half_system_size) / histogram_bin)
    if (ix >= 0 and ix < number_of_bins):
        histogram_x[ix] += 1.0
    # histogram of final y positions
    iy = math.floor((position_y[propagation_time] + half_system_size) / histogram_bin)
    if (iy >= 0 and iy < number_of_bins):
        histogram_y[iy] += 1.0
f.close()
print("Done, trajectories saved in", position_file, "!")

extension_file = 'squared_displacement.dat'
f = open(extension_file, 'w')
f.write('# Squared displacement in the plane averaged over different trajectories\n')
f.write('# Number of trajectories    = %d\n' % (number_of_paths))
f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
f.write('# Length of each trajectory = %d\n' % (propagation_time))
f.write('# Time Squared_displacement\n')
for i in range(propagation_time + 1):
    f.write('%9d %12.8g\n' % (i, r2[i] / number_of_paths))
f.close()
print("Done, squared displacements saved in", extension_file, "!")

histogram_file = 'final_position_x_distribution.dat'
f = open(histogram_file, 'w')
f.write('# Distribution of final positions in the x direction\n')
f.write('# Number of trajectories    = %d\n' % (number_of_paths))
f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
f.write('# Length of each trajectory = %d\n' % (propagation_time))
f.write('# x P(x)\n')
for i in range(number_of_bins):
    f.write('%12.8g %12.8g\n' % (
    (i + 0.5) * histogram_bin - half_system_size, histogram_x[i] / (number_of_paths * histogram_bin)))
f.close()
print("Done, distribution of final x positions saved in", histogram_file, "!")
histogram_file = 'final_position_y_distribution.dat'
f = open(histogram_file, 'w')
f.write('# Distribution of final positions in the y direction\n')
f.write('# Number of trajectories    = %d\n' % (number_of_paths))
f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
f.write('# Length of each trajectory = %d\n' % (propagation_time))
f.write('# y P(y)\n')
for i in range(number_of_bins):
    f.write('%12.8g %12.8g\n' % (
    (i + 0.5) * histogram_bin - half_system_size, histogram_y[i] / (number_of_paths * histogram_bin)))
f.close()
print("Done, distribution of final y positions saved in", histogram_file, "!")

if (cheat):
    diffusion_coefficient = 1.0 / (1.0 - math.exp(-0.5 * typical_scattering_angle ** 2))
    f = open('squared_displacement_analytic.dat', 'w')
    f.write('# Squared displacement in the plane computed analytically\n')
    f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
    f.write('# Length of each trajectory = %d\n' % (propagation_time))
    f.write('# Time Squared_displacement\n')
    for i in range(propagation_time + 1):
        f.write('%9d %12.8g\n' % (i, 2.0 * i * diffusion_coefficient))
    f.close()
    f = open('final_position_distribution_analytic.dat', 'w')
    f.write('# Typical scattering angle  = %g\n' % (typical_scattering_angle))
    f.write('# Length of each trajectory = %d\n' % (propagation_time))
    f.write('# x P(x)\n')
    for i in range(number_of_bins):
        x = (i + 0.5) * histogram_bin - half_system_size
        f.write('%12.8g %12.8g\n' % (x, math.exp(-0.5 * x * x / (diffusion_coefficient * propagation_time)) / math.sqrt(
            2.0 * math.pi * diffusion_coefficient * propagation_time)))
    f.close()

