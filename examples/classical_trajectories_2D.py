#!/usr/bin/python
from __future__ import print_function
import math
import sys
import numpy as np
from scipy.integrate import ode

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
# classical_trajectories_2D.py
# Author: Dominique Delande
# Release date: Mar, 5, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# ----------------------------------------------------------------
# This script models the classical motion of particles in a 2D random potential
# The potential is the sum of Gaussians with random amplitude
# (Gaussian distribution) and unit spatial extension
# ----------------------------------------------------------------
# The script can be ran with python 2 or 3. It requires numpy and scipy.
# It computes the classical propagation of particles in a 2D random potential.
# Each trajectory is computed by numerically integrating the Newton equations.
# The initial point is at the origin, the initial direction of propagation
# random. The initial velocity is calculated using that the total energy is the
# same for all trajectories.
# The script propagates "number_of_trajectories" trajectories during a time
# "propagation_time".
# The disordered potential is created by superimposing elementary Gaussian
# potentials (with sigma=1) with maximum value randomly chosen in a
# normal distribution (sigma=1) located at random positions in a square of size
# "system_size" with density "density_of_scatterers".
# No seed in the random number generation, but this can be easily added (line is
# commented out).
#
# The script outputs 5 files:
# * potential.dat which is the potential computed on a square grid. It can be
# visualized using e.g. matplotlib. At low density_of_scatterers, it is a random
# distribution of wells and bumps.
# * trajectories.dat which contains the (x,y) coordinates of the first 10
# trajectories.
# * squared_displacement.dat, which contains <x^2+y^2> (averaged over all
# trajectories) vs. time
# * final_position_x_distribution.dat and final_position_y_distribution.dat which
# contain the distribution of final positions. This is an histogram covering
# the range [-"system_size"/2,-"system_size"/2]. There are 50 bins, so this makes
# sense only if there are much more trajectories.
#
# The theoretical predictions for the squared_displacement and the final_position
# distributions are the following:
# * <r^2(t)> = 2Dt where D depends on the density of scatterers and the energy.
# * P(x) and P(y) are Gaussians with sigma=sqrt(Dt).
#
# In order to obtain a decent disordered potential, a good value is
# density_of_scatterers=0.2. Then, the distance between neighboring scatterers is
# of the order of 2, that is comparable to the size of the scatterers.
# ------------------------------------------------------
# classical_trajectories_2D.py
# Read parameters from the input line:
if len(sys.argv) != 6:
    print(
        'Usage (5 parameters): classical_trajectories_2D.py propagation_time density_of_scatterers system_size energy number_of_trajectories')
    print('       Each scatterer creates a Gaussian potential of size (sigma) = 1 ')
    print('       density_of_scatterers controls the average distance between scatterers')
    print('          if small, basically non overlapping scatterers')
    print('          use typically density_of_scatterers = 0.2')
    print('       Typical values for numerical parameters are')
    print(
        '       propagation_time=40, number_of_trajectorie=10, system_size=120, to see few multiple scattering trajectories')
    print('       propagation_time=40, number_oftrajectories=1000, system_size=120, to compute an histogram')
    print('                    of the final position distribution or compute the diffusion coefficient')
    sys.exit()
# Typical values for numerical parameters are
# propagation_time=100, number_of_paths=10 to see few multiple scattering paths
# propagation_time=100, number_of_paths=10000 to compute an histogram of the
# final position distribution or compute the diffusion coefficient
propagation_time = float(sys.argv[1])
number_of_trajectories = int(sys.argv[5])
system_size = float(sys.argv[3])
density_of_scatterers = float(sys.argv[2])
energy = float(sys.argv[4])
# The random number generator can be made reproducible by setting the seed
np.random.seed(42)

time_step = 1.0
number_of_bins = 50
number_of_pixels_for_potential = 100
number_of_printed_trajectories = 10

number_of_times = int(propagation_time / time_step)
number_of_scatterers = int(density_of_scatterers * system_size ** 2 + 0.5)
histogram_bin = system_size / number_of_bins
half_system_size = 0.5 * system_size
limit = half_system_size - 1.0

histogram_x = np.zeros(number_of_bins)
histogram_y = np.zeros(number_of_bins)
r2 = np.zeros(number_of_times + 1)
scatterer_position_x = np.zeros(number_of_scatterers)
scatterer_position_y = np.zeros(number_of_scatterers)
scatterer_amplitude = np.zeros(number_of_scatterers)

# Cheat code
# If system_size is negative, compute in addition the analytic results in some other files
if (system_size < 0.0):
    system_size = -system_size
    cheat = True
else:
    cheat = False


# This defines a single realization of the potential
def define_potential():
    scatterer_position_x = np.random.uniform(-half_system_size, half_system_size, number_of_scatterers)  # 随机数
    scatterer_position_y = np.random.uniform(-half_system_size, half_system_size, number_of_scatterers)
    scatterer_amplitude = np.random.normal(size=number_of_scatterers)  # np.random.normal(size,loc,scale): 给出均值为loc，标准差为scale的高斯随机数（场）.
    return (scatterer_position_x, scatterer_position_y, scatterer_amplitude)


def compute_potential(x, y):
    potential = np.sum(
        scatterer_amplitude * np.exp(-0.5 * (((scatterer_position_x - x) ** 2) + (scatterer_position_y - y) ** 2)))
    return potential


def compute_force(x, y):
    force_x = np.sum((x - scatterer_position_x) * scatterer_amplitude * np.exp(
        -0.5 * (((scatterer_position_x - x) ** 2) + (scatterer_position_y - y) ** 2)))
    force_y = np.sum((y - scatterer_position_y) * scatterer_amplitude * np.exp(
        -0.5 * (((scatterer_position_x - x) ** 2) + (scatterer_position_y - y) ** 2)))
    return (force_x, force_y)


def my_time_derivative(t, y):
    (force_x, force_y) = compute_force(y[0], y[1])
    return [y[2], y[3], force_x, force_y]


solve = ode(my_time_derivative)
# Use the standard vode integrator with limited accuracy as it is not crucial here
solve.set_integrator('vode', atol=1.e-4)

f = open('trajectories.dat', 'w')
f.write('# Classical trajectories in the plane\n')
f.write('# There are %d different trajectories, each with the (x,y) coordinates of the scattering events\n' % (
    number_of_printed_trajectories))
f.write('# Duration of each trajectory = %d\n' % (propagation_time))
f.write('# System size = %g\n' % (system_size))
f.write('# Density of scatterers = %g\n' % (density_of_scatterers))
for i_trajectory in range(number_of_trajectories):
    # Determine a potential sufficiently low to be able a particle at the origin
    # with the prescribed energy
    # If the generated potential does not work, throw it away and generate a new one

    while True:
        (scatterer_position_x, scatterer_position_y, scatterer_amplitude) = define_potential()
        if compute_potential(0., 0.) < energy:
            break

    # The potential is now chosen
    # If first trajectory, print it (for fun)
    if (i_trajectory == 1):
        g = open('potential.dat', 'w')
        x_bin = system_size / number_of_pixels_for_potential
        g.write('# %d %g\n' % (number_of_pixels_for_potential, x_bin))
        g.write('# %d %g\n' % (number_of_pixels_for_potential, x_bin))
        for i in range(number_of_pixels_for_potential):
            x = -half_system_size + (i + 0.5) * x_bin
            for j in range(number_of_pixels_for_potential):
                y = -half_system_size + (j + 0.5) * x_bin
                g.write("%g %g %g\n" % (y, x, compute_potential(x, y)))
        g.close()

    # determine initial condition with random velocity direction
    velocity = math.sqrt(2.0 * (energy - compute_potential(0., 0.)))
    angle = np.random.uniform(0.0, 2.0 * math.pi)
    y_initial = [0., 0., velocity * math.cos(angle), velocity * math.sin(angle)]
    if (i_trajectory < number_of_printed_trajectories):
        f.write('%g %g\n' % (y_initial[0], y_initial[1]))
    #  y=y_initial
    #  print('Initial energy =',0.5*(y[2]**2+y[3]**2)+compute_potential(y[0],y[1]))
    dt = time_step
    solve.set_initial_value(y_initial, 0.0)
    i_times = 0
    while solve.successful() and solve.t < propagation_time:
        y = solve.integrate(solve.t + dt)
        # Accumulate the squared displacement
        if ((abs(y[0]) > limit) or (abs(y[1]) > limit)):
            print('Warning, the trajectory is too close to the system edge! Increase system_size')
        i_times += 1
        r2[i_times] += y[0] ** 2 + y[1] ** 2
        # if in the first trajectories, print it
        if (i_trajectory < number_of_printed_trajectories):
            f.write('%g %g\n' % (y[0], y[1]))
    # if problem with energy conservation print a warning
    final_energy = 0.5 * (y[2] ** 2 + y[3] ** 2) + compute_potential(y[0], y[1])
    if (abs(final_energy - energy) > 1.e-2):
        print('Warning, the final energy is', final_energy)
        print('   You should decrease atol')
    f.write('\n\n')
    # histogram of final x positions
    ix = math.floor((y[0] + half_system_size) / histogram_bin)
    if (ix >= 0 and ix < number_of_bins):
        histogram_x[ix] += 1.0
    # histogram of final y positions
    iy = math.floor((y[1] + half_system_size) / histogram_bin)
    if (iy >= 0 and iy < number_of_bins):
        histogram_y[iy] += 1.0
    print("Done trajectory", i_trajectory)
f.close()

f = open('squared_displacement.dat', 'w')
f.write('# Squared displacement in the plane averaged over different trajectories\n')
f.write('# Number of trajectories    = %d\n' % (number_of_trajectories))
f.write('# Duration of each trajectory = %d\n' % (propagation_time))
f.write('# System size = %g\n' % (system_size))
f.write('# Density of scatterers = %g\n' % (density_of_scatterers))
f.write('# Time Squared_displacement\n')
for i_times in range(number_of_times + 1):
    f.write('%g %g\n' % (i_times * time_step, r2[i_times] / number_of_trajectories))
f.close()

histogram_file = 'final_position_x_distribution.dat'
f = open(histogram_file, 'w')
f.write('# Distribution of final positions in the x direction\n')
f.write('# Number of trajectories    = %d\n' % (number_of_trajectories))
f.write('# Duration of each trajectory = %d\n' % (propagation_time))
f.write('# System size = %g\n' % (system_size))
f.write('# Density of scatterers = %g\n' % (density_of_scatterers))
f.write('# x P(x)\n')
for i in range(number_of_bins):
    f.write('%12.8g %12.8g\n' % (
    (i + 0.5) * histogram_bin - half_system_size, histogram_x[i] / (number_of_trajectories * histogram_bin)))
f.close()
histogram_file = 'final_position_y_distribution.dat'
f = open(histogram_file, 'w')
f.write('# Distribution of final positions in the y direction\n')
f.write('# Number of trajectories    = %d\n' % (number_of_trajectories))
f.write('# Duration of each trajectory = %d\n' % (propagation_time))
f.write('# System size = %g\n' % (system_size))
f.write('# Density of scatterers = %g\n' % (density_of_scatterers))
f.write('# y P(y)\n')
for i in range(number_of_bins):
    f.write('%12.8g %12.8g\n' % (
    (i + 0.5) * histogram_bin - half_system_size, histogram_y[i] / (number_of_trajectories * histogram_bin)))
f.close()