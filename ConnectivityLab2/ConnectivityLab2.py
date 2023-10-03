from Graph import Graph
from collections import deque
import numpy as np



def transpose_graph(adj_matrix):
    num_nodes = len(adj_matrix)
    transposed = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            transposed[i][j] = adj_matrix[j][i]
    
    return transposed

def dfs(node, graph, visited, stack):
    visited[node] = True
    for neighbor, connected in enumerate(graph[node]):
        if not visited[neighbor] and connected:
            dfs(neighbor, graph, visited, stack)
    stack.append(node)

def kosaraju(graph):
    num_nodes = len(graph)
    visited = [False] * num_nodes
    stack = []
    
    # Первый этап: Обход графа и заполнение стека
    for node in range(num_nodes):
        if not visited[node]:
            dfs(node, graph, visited, stack)

    # Создание обратного графа
    transposed = transpose_graph(graph)

    visited = [False] * num_nodes
    strongly_connected_components = []

    # Второй этап: Обход графа в порядке убывания времени окончания обратного обхода
    while stack:
        node = stack.pop()
        if not visited[node]:
            component = []
            dfs(node, transposed, visited, component)
            strongly_connected_components.append(sorted(component))
    
    return strongly_connected_components

# # Пример матрицы смежности ориентированного графа
# adjacency_matrix = np.array(g.get_adjacency_matrix())

# # Вызов функции Косарайю для поиска компонент сильной связности
# strongly_connected_components = kosaraju(adjacency_matrix.tolist())

# # Вывод результатов
# for i, component in enumerate(strongly_connected_components):
#     print(f"Компонента {i + 1}: {component}")





def find_connected_components_in_graph(adj_matrix):

        
        num_nodes = len(adj_matrix)
        visited = [False] * num_nodes  # Список посещенных вершин
        components = []  # Список компонент связности

        # Функция для обхода в ширину
        def bfs(node):
            nonlocal visited
            component = []
            queue = deque([node])

            while queue:
                curr_node = queue.popleft()
                if not visited[curr_node]:
                    visited[curr_node] = True
                    component.append(curr_node)

                    for neighbor in range(num_nodes):
                        if adj_matrix[curr_node][neighbor] and not visited[neighbor]:
                            queue.append(neighbor)

            return component

            # Проход по всем вершинам графа
        for node in range(num_nodes):
            if not visited[node]:
                component = bfs(node)
                components.append(sorted(component))

            # Определение связности графа
        is_connected = len(components) == 1

        return is_connected, components

        
def directed_to_undirected(adj_matrix):
    
    # Преобразуем направленную матрицу смежности в неориентированную
    undirected_matrix = np.array(adj_matrix) + np.array(adj_matrix).T
    
    # Ограничим значения в матрице до 1 (если есть связь между вершинами, то 1, иначе 0)
    undirected_matrix[undirected_matrix > 1] = 1
    
    return undirected_matrix


    
def connected_components(graph : Graph):
    if not graph.is_directed:
        is_connected, components = find_connected_components_in_graph(graph.get_adjacency_matrix())
        num_components = len(components)
    
        print("Связность графа:", is_connected)
        print("Количество компонент связности:", num_components)
        print("Состав компонент связности:")
        for component in components:
            print(sorted(component))
    else:
        undirected_matrix = directed_to_undirected(graph.get_adjacency_matrix())
        # print(undirected_matrix)


        is_connected, components = find_connected_components_in_graph(undirected_matrix)
        num_components = len(components)

        print("Связность графа:", is_connected)
        print("Количество компонент связности:", num_components)
        print("Состав компонент связности:")
        print(components)

        # Пример матрицы смежности ориентированного графа
        adjacency_matrix = np.array(graph.get_adjacency_matrix())
        
        # Вызов функции Косарайю для поиска компонент сильной связности
        strongly_connected_components = kosaraju(adjacency_matrix.tolist())
        
        # Вывод результатов
        for i, component in enumerate(strongly_connected_components):
            print(f"Компонента {i + 1}: {component}")

            