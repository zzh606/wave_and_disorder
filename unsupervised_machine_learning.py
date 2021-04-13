#############
## 复现文章： Identifying topological order through unsupervised machine learning
#############
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map as dm
from XY_model import XYSystem
import random
import seaborn as sns


N = int(4**2)  # N 是模型中的粒子数
m = 100  # m 是XY模型的realization数，m越大效果越明显
SS = np.zeros((m, N))
S = np.zeros((m, N, 2))
config_matrix = np.zeros((m, int(np.sqrt(N)), int(np.sqrt(N))))
eps = 0.007  # Gaussian kernel参数，0.007
t = 50  # step次数
standard_dev = 0.3  # 标准差，0.3


def generate_theta(N, dev):
    theta = np.linspace(0, 0, N)
    v = random.randint(0, 1)
    for i in range(1, N+1):
        theta[i-1] = 2 * np.pi * v * i / N
        theta[i-1] += random.gauss(0, dev)
    theta += random.uniform(0, 2*np.pi)
    return theta


def matrixPow(Matrix,n):
    if(type(Matrix)==list):
        Matrix=np.array(Matrix)
    if(n==1):
        return Matrix
    else:
        return np.matmul(Matrix,matrixPow(Matrix,n-1))


# 1D XY自旋模型
for i in range(m):
    SS[i] = generate_theta(N, standard_dev)

# 2D XY自旋模型
# for i in range(m):
#     xy_system_1 = XYSystem(temperature=0.89, width=int(np.sqrt(N)))
#     spin_config = xy_system_1.spin_config
#     config_matrix[i] = xy_system_1.list2matrix(spin_config)
# SS = np.round(spin_config / (2 * np.pi / (N ** 2)))

S[:, :, 0] = np.cos(SS)
S[:, :, 1] = np.sin(SS)

K = np.zeros((m, m))  # Gaussian kernel
KK = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        # 这里ord=2 和ord=None结果不同？？？
        K[i, j] = np.exp(-np.linalg.norm(S[i] - S[j]) / (2 * N * eps))
        # K[i, j] = np.exp(-np.sqrt(np.sum((S[i] - S[j]) ** 2)) / (2 * N * eps))
        # K[i, j] = np.exp(-1 + np.sum(S[i]*S[j]) / N)

# K的热力图
# sns.heatmap(K, cmap="RdBu_r")
# plt.show()

z = np.sum(K, axis=1)
P = np.zeros((m, m))
for i in range(m):
    P[i] = K[i] / z[i]

A = np.zeros((m, m))  # similar matrix
for i in range(m):
    for j in range(m):
        A[i, j] = P[i, j] * np.sqrt(z[i] / z[j])

(eigenvalues, eigenstates) = np.linalg.eigh(A)
e = -eigenvalues
idx = e.argsort()  # 降序排列
eigenvalues = eigenvalues[idx]
eigenstates = eigenstates[:, idx]

# phi(1)直方图
# plt.hist(eigenstates[:, 1] * np.sqrt(m), bins=200)
# plt.show()

# P的本征值分布
x = np.linspace(0, np.size(eigenvalues), np.size(eigenvalues))
plt.scatter(x, eigenvalues)
print(eigenvalues[0:30])
plt.show()

Dt = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        temp = eigenvalues ** (2 * t) * (eigenstates[i] - eigenstates[j]) ** 2
        Dt[i, j] = np.sum(temp[1:])

# print(Dt)

