#!/usr/bin/python
from __future__ import print_function
import math
import random
import sys
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import copy
import os
import seaborn as sns
from scipy import io as scio

__license__ = "GPL version 2 or later"
__version__ = "1.1"

# compute_disorder_IPR_2D.py
# Authors: Zehuan Zheng
# Release date: April, 2, 2017
# License: GPL2 or later
# Tested with Python v2 and Python v3
# -----------------------------------------------------------------------------------------
# This script models localization in the 2d Anderson model with box disorder,
# i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
# The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
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

L = 16
nr = 10
t0 = 1 # 3.14e-2  # hopping系数，由wannier函数决定
alpha = 1 # 1.1454  # hopping系数，由wannier函数决定
rho_max = 0.5
eps_max = 0.5
scale = 1
path = 'data_test'


def generate_disorder_positions(L, scale, rho, eps):
    pos = np.zeros((L, L, 2))
    delta_pos = np.zeros((L, L, 2))
    for j in range(L):
        for i in range(L):
            delta_x = rho * np.cos(random.uniform(0, 2*np.pi)) + random.uniform(-eps, eps)
            delta_y = rho * np.sin(random.uniform(0, 2*np.pi)) + random.uniform(-eps, eps)
            pos[i, j, :] = [i * scale + delta_x, j * scale + delta_y]
            delta_pos[i, j, :] = [delta_x, delta_y]

    return [pos, delta_pos]


def generate_uniformly_random_point_set(N, x_range=None, y_range=None):
    if x_range is None:
        x_range = (0, 1)
    if y_range is None:
        y_range = (0, 1)
    pos = np.zeros((N, 2))

    for i in range(N):
        pos[i] = [x_range[0] + (x_range[1] - x_range[0]) * random.uniform(x_range[0], x_range[1]), y_range[0] + (y_range[1] - y_range[0]) * random.uniform(y_range[0], y_range[1])]

    return pos


def compute_tij(x, y, t0, alpha):
    dist = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    t = t0 * np.exp(-alpha * dist)
    return t


# Generate the Hamiltonian matrix for one realization of the random disorder
def generate_hamiltonian(position, t0, alpha, on_site_energy):
    L = np.size(position, 0)
    H = np.zeros((L * L, L * L))

    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    if i1 == i2 and j1 == j2:
                        tij = on_site_energy
                    else:
                        tij = compute_tij(position[i1, j1], position[i2, j2], t0, alpha)
                    H[i1 * L + j1, i2 * L + j2] = tij
                    H[i2 * L + j2, i1 * L + j1] = tij
    return H


def compute_IPR(eigenstates):
    IPR = 1.0 / np.sum(eigenstates ** 4, axis=0) / np.size(eigenstates, 0)
    return IPR


def generate_multicolor_image(delta_pos):
    x_image = delta_pos[:, :, 0]
    y_image = delta_pos[:, :, 1]

    f, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    sns.heatmap(x_image, ax=ax1, vmin=-1, vmax=1, cmap="Reds")
    sns.heatmap(y_image, ax=ax2, vmin=-1, vmax=1, cmap="Blues")
    plt.show()
    return [x_image, y_image]


def save_data(file_point, x_image, y_image, IPR):
    np.set_printoptions(linewidth=5000)
    file_point.write(str(x_image.flatten()) + ',')
    file_point.write(str(y_image.flatten()) + ',')
    file_point.write(str(IPR.flatten()) + '\n')


if __name__ == '__main__':
    ## 测试
    [pos, delta_pos] = generate_disorder_positions(L, scale=2, rho=0.5, eps=0.5)
    generate_multicolor_image(delta_pos)
    plt.figure(1)
    for j in range(L):
        for i in range(L):
            plt.scatter(pos[i, j, 0], pos[i, j, 1], c='g')
    my_x_ticks = np.arange(0, 16*2, 1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_x_ticks)
    plt.grid(True) #增加网格，也可以改变网格的线形，颜色
    plt.grid()
    plt.grid(color='black')
    plt.grid(linewidth='0.3')
    plt.show()
    exit()
    energy_levels = np.zeros((L * L, nr))
    wavefunc_num = np.linspace(0, L*L-1, L*L)
    rho_list = np.linspace(0, rho_max*(1-1/50), 50)
    eps_list = np.linspace(0, eps_max*(1-1/50), 50)
    x_image_arr = np.zeros((nr, L, L))
    y_image_arr = np.zeros((nr, L, L))
    IIPR_arr = np.zeros((nr, L*L))

    ## 发现已有数据
    rho_eps_exist = []
    for root, dirs, files in os.walk(path, topdown=True):
        for f in files:
            f = f.replace('.mat', '')
            f = f.split('-')
            rho_eps_exist.append([float(f[3].split('=')[1]), float(f[4].split('=')[1])])

    for rho in rho_list:
        for eps in eps_list:
            if [rho, eps] in rho_eps_exist:
                continue

            filename = 'images_IIPR-L='+str(L)+'-scale='+str(scale)+'-rho='+str(rho)+'-eps='+str(eps)+'.mat'
            print('Write to ' + filename + ': ', end='')
            # f = open(path + '\\' + filename, 'a')
            for ir in range(nr):
                [pos, delta_pos] = generate_disorder_positions(L, scale=scale, rho=rho, eps=eps)
                [x_image_arr[ir, :, :], y_image_arr[ir, :, :]] = generate_multicolor_image(delta_pos, rho, eps)
                H = generate_hamiltonian(pos, t0, alpha, on_site_energy=1.0)
                (energy_levels[:, ir], eigenstates) = linalg.eigh(H)
                IIPR = compute_IPR(eigenstates)
                idx = IIPR.argsort()
                IIPR_arr[ir, :] = copy.copy(IIPR[idx])
                # save_data(f, x_image, y_image, IIPR)
                print('.', end='')
            scio.savemat(path + '\\' + filename,
                         {'x_image_arr': x_image_arr, 'y_image_arr': y_image_arr, 'IIPR_arr': IIPR_arr})

            print('')
            # f.close()

    # IPR_aver = np.sum(IPR, 1) / nr
    # plt.figure(1)
    # plt.scatter(wavefunc_num, IPR_aver)
    # plt.show()

