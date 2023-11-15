from Graph import Graph


# поиск остовного дерева графа
# Результатом является список рёбер графа, входящих в остовное дерево и суммарный вес дерева
# крускала

def undirected_adjacency_matrix(adjacency_matrix):
    # Получаем размерность матрицы
    n = len(adjacency_matrix)
    
    # Создаем нулевую матрицу с размером n x n для соотнесенного графа
    undirected_matrix = [[0] * n for _ in range(n)]
    
    # Заполняем соотнесенную матрицу на основе направленной
    for i in range(n):
        for j in range(i, n):
            undirected_matrix[i][j] = adjacency_matrix[i][j] or adjacency_matrix[j][i]
            undirected_matrix[j][i] = undirected_matrix[i][j]
    
    return undirected_matrix



def find(parent, node):
    if parent[node] == node:
        return node
    return find(parent, parent[node])

def union(parent, rank, u, v):
    root_u = find(parent, u)
    root_v = find(parent, v)

    if rank[root_u] < rank[root_v]:
        parent[root_u] = root_v
    elif rank[root_u] > rank[root_v]:
        parent[root_v] = root_u
    else:
        parent[root_v] = root_u
        rank[root_u] += 1

def kruskal(graph):
    num_vertices = len(graph)
    edges = []

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    edges.sort(key=lambda x: x[2])
    parent = list(range(num_vertices))
    rank = [0] * num_vertices
    mst = []
    mst_weight = 0

    for edge in edges:
        u, v, weight = edge
        if find(parent, u) != find(parent, v):
            mst.append((u, v, weight))  # Включаем вес ребра в остовное дерево
            mst_weight += weight
            union(parent, rank, u, v)

    return mst, mst_weight



# Прима


import heapq
import numpy as np

def matrix_to_adjacency_list(matrix):
    graph = {}
    num_nodes = len(matrix)
    
    for i in range(num_nodes):
        graph[i] = []
        for j in range(num_nodes):
            if matrix[i][j] != 0:
                graph[i].append((j, matrix[i][j]))
    
    return graph

def prim(adj_matrix):
    matrix = np.array(adj_matrix)
    graph = matrix_to_adjacency_list(matrix)
    
    minimum_spanning_tree = []
    num_nodes = len(matrix)
    visited = set()
    
    # Выбираем начальную вершину (можно выбрать любую)
    start_node = 0
    visited.add(start_node)
    
    # Создаем приоритетную очередь для хранения ребер с их весами
    edge_heap = [(weight, start_node, neighbor) for neighbor, weight in graph[start_node]]
    heapq.heapify(edge_heap)
    
    while edge_heap:
        # Извлекаем ребро с минимальным весом
        weight, node1, node2 = heapq.heappop(edge_heap)
        
        # Если вершина node2 еще не посещена, добавляем ребро в остовное дерево
        if node2 not in visited:
            visited.add(node2)
            minimum_spanning_tree.append((node1, node2, weight))
            
            # Добавляем соседние ребра вершины node2 в приоритетную очередь
            for neighbor, edge_weight in graph[node2]:
                if neighbor not in visited:
                    heapq.heappush(edge_heap, (edge_weight, node2, neighbor))
    
    total_weight = sum(weight for _, _, weight in minimum_spanning_tree)
    
    return minimum_spanning_tree, total_weight




# from Graph import *

matrix_file = "task4/matrix_t4_010.txt"
import time
g = Graph(matrix_file, "-m")


if not g.is_directed():
    start = time.time()
    mst_edges, mst_weight = kruskal(g.get_adjacency_matrix())
    end = time.time()
    res = end - start
    print(res)
    print("Остовное дерево:", mst_edges)
    print("Суммарный вес дерева:", mst_weight)
elif g.is_directed():
    # print(undirected_adjacency_matrix(g.get_adjacency_matrix()))
    mst_edges, mst_weight = kruskal(undirected_adjacency_matrix(g.get_adjacency_matrix()))
    print("Остовное дерево:", mst_edges)
    print("Суммарный вес дерева:", mst_weight)
    

g.get_graph()


# print(g._Graph__adjacency_matrix)





# from Graph import *

matrix_file = "task4/matrix_t4_010.txt"

g = Graph(matrix_file, "-m")

if not g.is_directed():
    start = time.time()

    minimum_spanning_tree, total_weight = prim(g.get_adjacency_matrix())
    end = time.time()
    res = end - start
    print(res)
    print("Остовное дерево:", minimum_spanning_tree)
    print("Суммарный вес дерева:", total_weight)
elif g.is_directed():
    minimum_spanning_tree, total_weight = prim(undirected_adjacency_matrix(g.get_adjacency_matrix()))
    print("Остовное дерево:", minimum_spanning_tree)
    print("Суммарный вес дерева:", total_weight)
    

g.get_graph()




class AlgBoruvka:
    def __init__(self, graph):
        self._graph = graph
        self._matrix = undirected_adjacency_matrix(graph.get_adjacency_matrix())
        self._matrix_len = len(self._matrix)

    # нахождение минимального остовного дерева
    def spanning_tree(self):
        tree = []
        edges = self._graph._Graph__edges
        # edges = self._graph.list_of_edges()
        parents, size = self.__init_DSU()  # инициализация DSU

        # количество компонент связности, изначально каждая вершина это отдельная компонента
        components = self._matrix_len

        # индекс минимального ребра из каждой компоненты связности
        min_edge = [-1 for i in range(self._matrix_len)]

        # пока не останется одна компонента связности
        while components != 1:
            # изнаально минимальное ребро для каждой компоненты равно -1
            for i in range(self._matrix_len):
                min_edge[i] = -1

            # перебираем все ребра 
            for edge in edges:
                # если ребро соединяет одинаковые компоненты связности - пропускаем
                if self.__root(edge[0], parents) == self.__root(edge[1], parents):
                    continue

                # находим лидера вершины v из ребра (v, u) и если минимальное ребро не найдено 
                # или вес просматриваемого ребра меньше веса минимального ребра - 
                # запоминаем индекс минимального ребра для leader_v вершины
                leader_v = self.__root(edge[0], parents)
                if min_edge[leader_v] == -1 or edge[2] < min_edge[leader_v][2]:
                    min_edge[leader_v] = edge

                # аналогично для лидера вершины u из ребра (v, u)
                leader_u = self.__root(edge[1], parents)
                if min_edge[leader_u] == -1 or edge[2] < min_edge[leader_u][2]:
                    min_edge[leader_u] = edge

            # если минимальное ребро найдено - объединяем компоненты, добавляем ребро в дерево
            # а также количество компонент связности уменьшаем на единицу
            for i in range(self._matrix_len):
                if min_edge[i] != -1 and self.__union(min_edge[i][0], min_edge[i][1], parents, size):
                    # tree.append([min_edge[i][0] + 1, min_edge[i][1] + 1, min_edge[i][2]])
                    tree.append([min_edge[i][0], min_edge[i][1], min_edge[i][2]])
                    components -= 1

        # подсчет веса полученного дерева
        tree_weight = sum((tree[i][2] for i in range(len(tree))))

        return tree, tree_weight

    # DSU аналогичное алгоритму Краскала с эвристиками сжатия путей и весов деревьев
    def __init_DSU(self):
        p = [i for i in range(self._matrix_len)]
        s = [1] * self._matrix_len
        return p, s

    def __root(self, vertice, parent):
        if parent[vertice] != vertice:
            parent[vertice] = self.__root(parent[vertice], parent)
        return parent[vertice]

    def __union(self, aa, bb, parent, size):
        a = self.__root(aa, parent)
        b = self.__root(bb, parent)
        if a == b:
            return False
        elif size[a] > size[b]:
            parent[b] = a
            size[a] += size[b]
        else:
            parent[a] = b
            size[b] += size[a]
        return True





start = time.time()

gr = AlgBoruvka(g)
print(gr.spanning_tree())
end = time.time()
res = end - start
# print(g._Graph__edges)

print(res)
