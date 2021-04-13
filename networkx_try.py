import networkx as nx
from matplotlib import pyplot as plt

G = nx.Graph() # 创建空图
G.add_edge(1, 2, length = 10) # 为边 (1,2) 添加属性 length = 10
G.add_edge(1, 3, length = 100) # 为边 (1,3) 添加属性 weight = 20
G.add_edge(2, 3, length = 15) # 为边 (2,2) 添加属性 capacity = 15
nx.draw(G,with_labels=True) # 画出图

plt.show()