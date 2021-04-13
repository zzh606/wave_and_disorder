"""
复现文章： Unsupervised Manifold Clustering of Topological Phononics
"""
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from compute_disorder_IPR_2D import generate_disorder_positions
import cmath
import networkx as nx  ## 安装pygraphviz  用conda install -c alubbock pygraphviz
import pygraphviz as pyg


def compute_tij(x, y, t0, alpha):
    dist = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    t = t0 * np.exp(-alpha * dist)
    return t


def generate_hamiltonian_from_voronoi(voronoi, t0, alpha, on_site_energy):
    vertices = voronoi.vertices
    ridges = voronoi.ridge_vertices
    N = len(ridges)   # 连接数
    L = len(vertices)  # 粒子数
    H = np.zeros((L, L))

    for i in range(N):
        x = ridges[i][0]
        y = ridges[i][1]
        if x == -1 or y == -1:
            continue
        tij = compute_tij(vertices[x], vertices[y], t0, alpha)
        # tij = 1  # 假设往各个方向传递的概率相等
        H[x, y] = tij
        H[y, x] = tij
    return H  # H按照vertices顺序排列


def generate_1D_SSH_Hamitonian(L, k0, m):
    H = np.zeros((L, L))
    delta = random.uniform(0, 1)

    for i in range(L-1):
        ip1 = (i + 1) % L  # 周期边界条件
        if i == 0:
            H[i, i] = -k0 * delta
        else:
            H[i, i] = -2 * k0

        if i % 2 == 0:
            H[i, ip1] = k0 * delta
            H[ip1, i] = k0 * delta
        else:
            H[i, ip1] = k0 * (1 - delta)
            H[ip1, i] = k0 * (1 - delta)
    H[L - 1, L - 1] = -k0 * delta
    # H[L - 2, L - 1] = 0
    return H * -m, delta


def generate_1D_SSH_with_elastic_constants_Hamitonian(L, k0, m):
    H = np.zeros((L, L))
    delta_k = random.uniform(-0.75, 0.75)
    iota = [random.uniform(0, 1) for _ in range(L)]
    for i in range(L-1):
        ip1 = (i + 1) % L  # 周期边界条件

        if i % 2 == 0:
            if i == 0:
                H[i, i] = -k0 - iota[i] * delta_k
            else:
                H[i, i] = -2 * k0 - (iota[i] - iota[i-1]) * delta_k
            H[i, ip1] = k0 + iota[i] * delta_k
            H[ip1, i] = k0 + iota[i] * delta_k
        else:
            H[i, i] = -2 * k0 - (iota[i - 1] - iota[i]) * delta_k
            H[i, ip1] = k0 - iota[i] * delta_k
            H[ip1, i] = k0 - iota[i] * delta_k
    H[L - 1, L - 1] = -k0 - iota[L-2] * delta_k
    return H * -m, delta_k


def generate_projector(H, wc):
    (eigenvalues, phi) = np.linalg.eigh(H)
    L = np.size(H, 0)

    Pi = np.zeros((L, L))
    for l in range(L):
        if eigenvalues[l] <= wc ** 2:  # 20年PRL和18年NP的大于小于号不同
            Pi += np.outer(phi[:, l], phi[:, l])
    return Pi


def generate_projector_of_graph(H, graph, wc):
    (eigenvalues, phi) = np.linalg.eigh(H)
    # nodes = list(graph.nodes)
    # edges = graph.edges
    L = np.size(H, 0)

    # Pi = np.zeros((L, L))
    # for i in range(L):
    #     for j in range(L):
    #         for l in range(L):
    #             if eigenvalues[l] <= wc ** 2:  # 20年PRL和18年NP的大于小于号不同
    #                 Pi[i, j] += phi[i, l] * phi[j, l]

    Pi = np.zeros((L, L))
    for l in range(L):
        if eigenvalues[l] <= wc ** 2:  # 20年PRL和18年NP的大于小于号不同
            Pi += np.outer(phi[:, l], phi[:, l])

    return Pi


def generate_area_projector(H, area_vertices, wc):
    L = np.size(H, 0)
    (eigenvalues, phi) = np.linalg.eigh(H)

    # 补充材料Pij的算法，和投影算法等价
    # for j in range(L):
    #     for k in range(L):
    #         for l in range(L):
    #             if np.sqrt(eigenvalues[l]) > wc:
    #                 break
    #             P[i, j, k] += phi[i, k, l] * phi[i, j, l]

    # 投影算符的算法
    Pi = np.zeros((len(area_vertices), len(area_vertices)))
    for i in range(len(area_vertices)):
        for j in range(len(area_vertices)):
            for l in range(L):
                if eigenvalues[l] <= wc ** 2:  # 20年PRL和18年NP的大于小于号不同
                    #### 这里把subarea的坐标(area_vertices[i], area_vertices[j])映射到P的坐标(i, j)
                    Pi[i, j] += phi[area_vertices[i], l] * phi[area_vertices[j], l]  # 共轭何时取？
                    # Pi[i, j] += np.outer(phi[:, l], phi[:, l])[area_vertices[i], area_vertices[j]]  # 取出[area_vertices[i], area_vertices[j]]作为Pij的元素
    # print(Pi)

    # P的热力图
    # sns.heatmap(P[0], cmap="jet", vmin=0, vmax=1)
    # plt.show()
    return Pi


def calculate_voronoi_chern(P, position, vertices_num):
    pos_list = []
    nodes_list = []

    if type(position) is np.ndarray:
        for i in range(np.size(position, 0)):
            nodes_list.append(i)
            pos_list.append(position[i])
    else:
        for each in position:
            nodes_list.append(each)
            pos_list.append(position[each])

    pos_list = np.array(pos_list)
    pos_list = pos_list[vertices_num]
    x_min = np.min(pos_list[:, 0])
    x_max = np.max(pos_list[:, 0])
    y_min = np.min(pos_list[:, 1])
    y_max = np.max(pos_list[:, 1])
    central = [(x_min + x_max) / 2, (y_min + y_max) / 2]

    area_A = []
    area_B = []
    area_C = []
    for i in range(np.size(pos_list, 0)):
        each = pos_list[i]
        angle = cmath.polar(complex(each[0] - central[0], each[1] - central[1]))[1]
        if angle < 0:
            angle += 2 * np.pi
        if 0 <= angle < np.pi * 2 / 3:
            area_A.append(i)
        elif np.pi * 2 / 3 <= angle < np.pi * 4 / 3:
            area_B.append(i)
        else:
            area_C.append(i)

    v = 0
    for j in range(len(area_A)):
        for k in range(len(area_B)):
            for l in range(len(area_C)):
                v += P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j]
    v *= 12 * np.pi
    return v


def split_voronoi_area(voronoi, C, xy_range, margin=0.01):
    # 这里的P是按照vertices的顺序分布
    vertices = voronoi.vertices
    ridges = voronoi.ridge_vertices
    x_min = xy_range[0] - (xy_range[1] - xy_range[0]) * margin
    x_max = xy_range[1] + (xy_range[1] - xy_range[0]) * margin
    y_min = xy_range[2] - (xy_range[3] - xy_range[2]) * margin
    y_max = xy_range[3] + (xy_range[3] - xy_range[2]) * margin

    x = np.linspace(x_min, x_max, C+1)
    y = np.linspace(y_min, y_max, C+1)

    vertice_sublist = []
    cnt = 0
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            vertice_sublist.append([])
            for k in range(len(vertices)):
                each = vertices[k]
                if x[i] < each[0] <= x[i + 1] and y[j] < each[1] <= y[j + 1]:
                    vertice_sublist[cnt].append(k)
            cnt += 1
    return vertice_sublist


def split_area_of_graph(graph, position, C, whole_area, margin=0.01):
    # 这里的P是按照vertices的顺序分布
    nodes = graph.nodes
    edges = graph.edges
    pos_list = []
    nodes_list = []

    if type(position) is np.ndarray:
        for i in range(np.size(position, 0)):
            nodes_list.append(i)
            pos_list.append(position[i])
    else:
        for each in position:
            nodes_list.append(each)
            pos_list.append(position[each])

    pos_list = np.array(pos_list)
    x_min = whole_area[0] - (whole_area[1] - whole_area[0]) * margin
    x_max = whole_area[1] + (whole_area[1] - whole_area[0]) * margin
    y_min = whole_area[2] - (whole_area[3] - whole_area[2]) * margin
    y_max = whole_area[3] + (whole_area[3] - whole_area[2]) * margin
    x = np.linspace(x_min, x_max, C+1)
    y = np.linspace(y_min, y_max, C+1)

    nodes_sublist = []
    cnt = 0
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            nodes_sublist.append([])
            for k in range(len(nodes_list)):
                each = nodes_list[k]
                if x[i] < position[each][0] <= x[i + 1] and y[j] < position[each][1] <= y[j + 1]:
                    nodes_sublist[cnt].append(k)
            cnt += 1

    return nodes_sublist


def classification_1D_SSH(P, delta_list):
    N = np.size(P, 0)
    K = np.zeros((N, N))  # Gaussian kernel
    for i in range(N):
        for j in range(N):
            # K[i, j] = np.exp(-np.linalg.norm(np.abs(P[i] - P[j]), ord=1) ** 2 / (2 * N * eps))
            K[i, j] = np.exp(-(np.sum(np.abs(P[i] - P[j]))) ** 2 / (2 * N * eps))
            # K[i, j] = np.exp(-np.sqrt(np.sum(np.abs(P[i] - P[j]) ** 2)) / (2 * L ** 2 * eps))

    Z = np.sum(K, axis=1)
    T = np.zeros((N, N))
    for i in range(N):
        T[i] = K[i] / Z[i]

    (eigenvalues, eigenstates) = np.linalg.eigh(T)
    e = -eigenvalues
    idx = np.argsort(e)  # 降序排列
    eigenvalues = eigenvalues[idx]
    eigenstates = eigenstates[:, idx]

    # phi(1)直方图
    # plt.hist(eigenstates[:, 1] * np.sqrt(m), bins=200)
    # plt.show()

    # P的本征值分布
    x = np.linspace(0, np.size(eigenvalues), np.size(eigenvalues))
    plt.scatter(x[0:9], eigenvalues[0:9])
    plt.show()

    # phi1-phi2 分布
    for i in range(N):
        plt.scatter(eigenstates[i, 1] * np.sqrt(N), eigenstates[i, 2] * np.sqrt(N), color=(delta_list[i], 0, 1-delta_list[i]))
    plt.xlabel('sqrt(N)*phi1')
    plt.ylabel('sqrt(N)*phi2')
    plt.show()


def classification_of_graph(P, N, eps, delta_m_list):
    K = np.zeros((N, N))  # Gaussian kernel
    for i in range(N):
        for j in range(N):
            K[i, j] = np.exp(-(np.sum(np.abs(P[i] - P[j]))) ** 2 / (2 * N * eps))

    Z = np.sum(K, axis=1)
    T = np.zeros((N, N))
    for i in range(N):
        T[i] = K[i] / Z[i]

    (eigenvalues, eigenstates) = np.linalg.eigh(T)
    e = -eigenvalues
    idx = np.argsort(e)  # 降序排列
    eigenvalues = eigenvalues[idx]
    eigenstates = eigenstates[:, idx]

    # phi(1)直方图
    # plt.hist(eigenstates[:, 1] * np.sqrt(m), bins=200)
    # plt.show()

    # P的本征值分布
    x = np.linspace(0, np.size(eigenvalues), np.size(eigenvalues))
    plt.scatter(x[0:9], eigenvalues[0:9])
    plt.show()

    # phi1-phi2 分布
    for i in range(N):
        plt.scatter(eigenstates[i, 1] * np.sqrt(N), eigenstates[i, 2] * np.sqrt(N), color=(delta_m_list[i], 0, 1-delta_m_list[i]))
    plt.xlabel('sqrt(N)*phi1')
    plt.ylabel('sqrt(N)*phi2')
    plt.show()


def generate_uniformly_random_point_set(N, x_range=None, y_range=None):
    if x_range is None:
        x_range = (0, 1)
    if y_range is None:
        y_range = (0, 1)
    pos = np.zeros((N, 2))

    for i in range(N):
        pos[i] = [x_range[0] + (x_range[1] - x_range[0]) * random.uniform(x_range[0], x_range[1]), y_range[0] + (y_range[1] - y_range[0]) * random.uniform(y_range[0], y_range[1])]

    return pos


def generate_voronoi(L):
    # position, _ = generate_disorder_positions(L, scale, rho, eps)
    position = generate_uniformly_random_point_set(L)

    vor = Voronoi(position)

    # fig = voronoi_plot_2d(vor)  # 绘图函数
    vertices = vor.vertices
    ridges = vor.ridge_vertices
    for i in range(len(ridges)):
        x = ridges[i][0]
        y = ridges[i][1]
        if x == -1 or y == -1:
            continue
        plt.plot([vertices[x][0], vertices[y][0]], [vertices[x][1], vertices[y][1]])
    plt.axis('equal')
    plt.axis([0, 1, 0, 1])
    plt.show()

    return vor


def generate_hexagonal_lattice(period, m0=1, delta_m_max=0.5):
    G = nx.Graph(directed=False)
    delta_m = random.choice([-delta_m_max, delta_m_max]) * m0
    G.add_node((0, 0), bias=m0+random.uniform(0, 1)*delta_m)
    sign = -1
    for n in range(period):
        for (q, r) in list(G.nodes()):
            G.add_node((q, r - 1), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q, r - 1), weight=1)
            G.add_node((q - 1, r), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q - 1, r), weight=1)
            G.add_node((q - 1, r + 1), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q - 1, r + 1), weight=1)
            G.add_node((q, r + 1), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q, r + 1), weight=1)
            G.add_node((q + 1, r - 1), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q + 1, r - 1), weight=1)
            G.add_node((q + 1, r), bias=m0+random.uniform(0, 1)*delta_m*sign)
            G.add_edge((q, r), (q + 1, r), weight=1)
        sign *= -1
    return G, delta_m


def generate_kagome_lattice(period, m0=1, k0=1):
    delta = random.uniform(0, 1)
    k1 = k0 * delta
    k2 = k0 * (1 - delta)

    G = nx.Graph(directed=False)
    for n in range(period):
        for k in range(n+1):
            # 0: 左下，1:右下，2:上
            G.add_edge((n, k, 0), (n, k, 1), weight=k1)
            G.add_edge((n, k, 1), (n, k, 2), weight=k1)
            G.add_edge((n, k, 2), (n, k, 0), weight=k1)

            if k < n:
                G.add_edge((n, k, 1), (n, k+1, 0), weight=k2)
            if n < period - 1:
                G.add_edge((n, k, 1), (n + 1, k + 1, 2), weight=k2)
                G.add_edge((n, k, 0), (n + 1, k, 2), weight=k2)
    for each in G.nodes:
        G.nodes[each]['bias'] = m0

    return G, delta


def draw_graph(G):
    # 节点位置
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')  # positions for all nodes
    # 首先画出节点位置
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos,width=6)
    # labels标签定义
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def generate_hamiltonian_of_graph(graph, t0, alpha):
    nodes = list(graph.nodes)
    edges = graph.edges
    L = len(nodes)
    H = np.zeros((L, L))

    for n1, n2 in edges:
        H[nodes.index(n1), nodes.index(n1)] = graph.nodes[n1]['bias']
        H[nodes.index(n2), nodes.index(n2)] = graph.nodes[n2]['bias']
        # H[nodes.index(n1), nodes.index(n2)] = t0 * np.exp(-alpha * graph.get_edge_data(n1, n2)['weight'])
        # H[nodes.index(n2), nodes.index(n1)] = t0 * np.exp(-alpha * graph.get_edge_data(n1, n2)['weight'])
        H[nodes.index(n1), nodes.index(n2)] = graph.get_edge_data(n1, n2)['weight']
        H[nodes.index(n2), nodes.index(n1)] = graph.get_edge_data(n1, n2)['weight']

    return H


"""
 参考 http://www.guanjihuan.com/archives/4622
 https://zhuanlan.zhihu.com/p/71638138 
"""
def density_of_state(H, E_range):
    plot_precision = 0.05  # 画图的精度
    Fermi_energy_array = np.arange(E_range[0] + plot_precision, E_range[1], plot_precision)  # 计算中取的费米能Fermi_energy组成的数组
    dim_energy = Fermi_energy_array.shape[0]   # 需要计算的费米能的个数
    total_DOS_array = np.zeros((dim_energy))   # 计算结果（总态密度total_DOS）放入该数组中
    dim = H.shape[0]   # 哈密顿量的维度
    i0 = 0
    for Fermi_energy in Fermi_energy_array:
        green = np.linalg.inv((Fermi_energy + 0.05j) * np.eye(dim) - H)   # 体系的格林函数
        total_DOS = -np.trace(np.imag(green)) / np.pi    # 通过格林函数求得总态密度
        total_DOS_array[i0] = total_DOS   # 记录每个Fermi_energy对应的总态密度
        i0 += 1
    sum_up = np.sum(total_DOS_array) * plot_precision    # 用于图像归一化
    # plt.plot(Fermi_energy_array, total_DOS_array/sum_up, '-')   # 画DOS(E)图像
    plt.bar(np.sqrt(Fermi_energy_array), total_DOS_array / sum_up, width=0.1)
    plt.xlabel('E')
    plt.ylabel('DOS')
    plt.axis([0, 2.5, 0.05, 1.5])
    plt.show()
    return np.sqrt(Fermi_energy_array), total_DOS_array / sum_up


def generate_voronoi_graph(L, t0, alpha):
    G = nx.Graph(directed=False)
    position = generate_uniformly_random_point_set(L)
    vor = Voronoi(position)
    # fig = voronoi_plot_2d(vor)  # 绘图函数
    vertices = vor.vertices
    ridges = vor.ridge_vertices
    pos = np.zeros((len(vertices), 2))
    for i in range(len(ridges)):
        x = ridges[i][0]
        y = ridges[i][1]
        if x == -1 or y == -1:
            continue
        # G.add_edge(vertices[x], vertices[y])
        G.add_node(x, pos=vertices[x], bias=0)
        G.add_node(y, pos=vertices[y], bias=0)
        G.add_edge(x, y, weight=compute_tij(vertices[x], vertices[y], t0, alpha))
        pos[x] = vertices[x]
        pos[y] = vertices[y]
    return G, pos


def main():
    L = 20  # 一个维度的长度
    C = 1  # 一个维度分成C个supercell
    N = 1000  # 模型的realization数
    H = np.zeros((N, L, L))
    H_voronoi = []
    phi = np.zeros((N, L, L))
    P = np.zeros((N, L, L))
    P_voronoi = []
    P_graph = []
    eps = 1
    k0 = 1
    m = 1
    wc = 1  # wc = np.sqrt(k0 / m)
    t0 = 1  # hopping系数，由wannier函数决定
    alpha = 1  # hopping系数，由wannier函数决定
    delta_list = []
    vor_list = []
    delta_m_list = []
    v = []

    for i in range(N):
        """ 1D SSH """
        # H[i], delta = generate_1D_SSH_with_elastic_constants_Hamitonian(L, k0, m)
        # P[i] = generate_projector(H[i], wc)
        # if delta >= 0:
        #     delta_list.append(1)
        # else:
        #     delta_list.append(0)
        # sns.heatmap(P[i], cmap="jet", vmin=0, vmax=0.6)
        # plt.show()

        """ voronoi. fun 1 """
        # vor = generate_voronoi(200)  #产生voronoi
        # vor_list.append(vor)
        # H_temp = generate_hamiltonian_from_voronoi(vor, t0, alpha, on_site_energy=1.0)
        # H_voronoi.append(H_temp)  # 产生每个area的H
        # area_vertices_num = split_voronoi_area(vor, C, [0, L, 0, L])  # 分割区域
        # P_voronoi.append([])
        # for j in range(len(area_vertices_num)):
        #     P_temp = generate_area_projector(H_voronoi[i], area_vertices_num[j])
        #     P_voronoi[i].append(P_temp)
        #     v_temp = calculate_voronoi_chern(P_temp, vor, area_vertices_num[j])
        #     v.append(v_temp)

        """ 2D 六角晶格，随机bias """
        # G, delta_m = generate_hexagonal_lattice(4)
        # if delta_m >= 0:
        #     delta_m_list.append(1)
        # else:
        #     delta_m_list.append(0)
        # pos = nx.nx_agraph.graphviz_layout(G, prog='neato')  # positions for all nodes

        """ Kagome lattice """
        # G, delta_m = generate_kagome_lattice(4)
        # if delta_m >= 0.26:
        #     delta_m_list.append(1)
        # else:
        #     delta_m_list.append(0)

        """  Voronoi. fun 2 """
        G, pos = generate_voronoi_graph(500, t0, alpha)

        """ 绘制graph图 """
        # pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        # for x, y in G.edges:
        #     print(G.get_edge_data(x, y))
        # nx.draw(G, pos, alpha=.75, with_labels=True)
        # plt.axis('equal')
        # plt.show()

        area_vertices_num = split_area_of_graph(G, pos, C, [0, 1, 0, 1])  # 分割区域
        H_graph = generate_hamiltonian_of_graph(G, t0, alpha)
        P_graph.append(generate_projector_of_graph(H_graph, G, wc))
        ##### 计算态密度DOS
        density_of_state(H_graph, [0, 10])

        ### 计算Chern number
        P_voronoi.append([])
        v.append([])
        for j in range(len(area_vertices_num)):
            P_temp = generate_area_projector(H_graph, area_vertices_num[j], wc)
            P_voronoi[i].append(P_temp)
            v_temp = calculate_voronoi_chern(P_temp, pos, area_vertices_num[j])
            v[i].append(v_temp)
        print(v[i])
        exit()
    # classification_1D_SSH(P, delta_list)
    classification_of_graph(P_graph, N, eps, delta_m_list)


if __name__ == '__main__':
    main()
