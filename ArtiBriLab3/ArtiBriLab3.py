from Graph import Graph

def find_articulation_points_and_bridges(adj_matrix):
    def dfs(u, parent, time):
        nonlocal visit_time
        visited[u] = True
        disc[u] = low[u] = time
        time += 1
        children = 0

        for v in range(len(adj_matrix[u])):
            if adj_matrix[u][v] == 1:
                if not visited[v]:
                    children += 1
                    dfs(v, u, time)
                    low[u] = min(low[u], low[v])

                    if low[v] > disc[u]:
                        bridges.append(sorted((u, v)))

                    if parent == -1 and children > 1:
                        articulation_points.add(u)
                    if parent != -1 and low[v] >= disc[u]:
                        articulation_points.add(u)
                else:
                    if v != parent:
                        low[u] = min(low[u], disc[v])

    n = len(adj_matrix)
    visited = [False] * n
    disc = [-1] * n
    low = [-1] * n
    articulation_points = set()
    bridges = []
    visit_time = 0

    for i in range(n):
        if not visited[i]:
            dfs(i, -1, visit_time)
    # print(articulation_points)
    return list(articulation_points), bridges





matrix_file = "task3/matrix_t3_010.txt"

g = Graph(matrix_file, "-m")
g.get_graph()

articulation_points, bridges = find_articulation_points_and_bridges(g._Graph__adjacency_matrix)

print(g.is_directed())

print("Мосты:", bridges)

print("Шарниры:", articulation_points)