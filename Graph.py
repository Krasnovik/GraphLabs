import networkx as nx
import numpy as np
from random import randint

class Graph:
    
    def __init__(self, file_path, file_type):
        self.__adjacency_matrix = None
        self.__adjacency_dict = None
        self.__edges = None
        self.__dist_matrix = None
        self.__colors = ['b','g','r','c','m','y','k']
        if file_type == '-m':
            self.__load_adjacency_matrix(file_path)
            self.__check_directed()
        elif file_type == '-l':
            self.__load_adjacency_dict(file_path)
            self.__adjacency_dict_to_matrix()
            self.__check_directed()
        elif file_type == '-e':
            self.__load_edges(file_path)
            self.__check_directed()
        else:
            raise ValueError("Invalid file type")
            
    def __check_directed(self): 
        for i in range(len(self.__adjacency_matrix)):
            for j in range(len(self.__adjacency_matrix)):
                # print(self.__adjacency_matrix[i][j], self.__adjacency_matrix[j][i])
                if self.__adjacency_matrix[i][j] != self.__adjacency_matrix[j][i]:
                    self.__is_directed = True
                    return
        self.__is_directed = False
        # return
#load funcs
    def __load_adjacency_matrix(self, file_path):
        with open(file_path, 'r') as file:
            matrix = []
            for line in file:
                row = [int(val) for val in line.strip().split()]
                matrix.append(row)
            self.__adjacency_matrix = matrix
            
            
    def __load_adjacency_dict(self, file_path):
        adjacency_dict = {}
        with open(file_path, 'r') as file:
            count = 0
            for line in file:
                adjacency_dict[count] = (list(map(int, line.strip().split())))
                count+=1
            self.__adjacency_dict = adjacency_dict
            
    def __load_edges(self, file_path):
        with open(file_path, 'r') as file:
            edge_list = []
#             weights = []
            for line in file:
                if len(line.strip().split()) == 2:
                    vertex1, vertex2 = map(int, line.strip().split())
                    edge_list.append([vertex1 - 1, vertex2 - 1])
                    self.__edges = edge_list
                    self.__convert_from_edges_to_adjacency_matrix()
                elif len(line.strip().split()) == 3:
                    vertex1, vertex2, weight = map(int, line.strip().split())
                    edge_list.append([vertex1 - 1, vertex2 - 1, weight])
#                     weights.append(weight)
                    self.__edges = edge_list
                    self.__convert_from_edges_to_adjacency_matrix()
                else:
                    raise ValueError("Invalid file contents")
            

    def __adjacency_dict_to_matrix(self):
        num_vertices = len(self.__adjacency_dict)
        matrix = [[0] * num_vertices for _ in range(num_vertices)]
        for vertex, neighbors in self.__adjacency_dict.items():
            for neighbor in neighbors:
                matrix[vertex][neighbor - 1] = 1
    
        self.__adjacency_matrix = matrix
        
    def __convert_from_edges_to_adjacency_matrix(self):
        num_vertices = max(max(edge[:2]) for edge in self.__edges) + 1
        matrix = [[0] * num_vertices for _ in range(num_vertices)]
        for edge in self.__edges:
            start, end = edge[:2]
            weight = edge[2] if len(edge) > 2 else 1
            matrix[start][end] = weight
    
        self.__adjacency_matrix = matrix

        
#get funcs
    def get_adjacency_matrix(self):
        return self.__adjacency_matrix
    def size(self):
        if self.__adjacency_matrix:
            return len(self.__adjacency_matrix)
        else:
            return 0
        
    def weight(self, vertex1, vertex2):
        if self.__adjacency_matrix:
            return self.__adjacency_matrix[vertex1][vertex2]
        else:
            return None
        
    def is_edge(self, vertex1, vertex2):
        if self.__adjacency_matrix:
                if self.__adjacency_matrix[vertex1][vertex2] != 0:
                    return True
                else:
                    return False
        else:
            return False
        
    def is_directed(self): #direc or not
        return self.__is_directed

    def get_graph(self):
        if self.__is_directed:
            G = nx.MultiDiGraph()
            G = nx.from_numpy_array(np.array(self.__adjacency_matrix), create_using=nx.MultiDiGraph())
            
            pos = nx.random_layout(G)
            nx.draw(G, 
                    pos,
                    with_labels=True, 
                    arrows = True, 
                    node_color=[self.__colors[randint(0, len(self.__colors)-2)] 
                                for node in G.nodes()], 
                    node_size=100, 
                    font_size=18, 
                    font_weight='bold', 
                    edge_color = [self.__colors[randint(0, len(self.__colors)-1)] 
                                  for edge in G.edges()], 
                    arrowsize=20, 
                    # arrowstyle='->', 
                    width=2.0)
        else:
            G = nx.Graph()
            G = nx.from_numpy_array(np.array(self.__adjacency_matrix))
            pos = nx.random_layout(G)
            nx.draw(G, 
                    pos,
                    with_labels=True, 
                    node_color=[self.__colors[randint(0, len(self.__colors)-1)] 
                                for node in G.nodes()], 
                    node_size=100, 
                    font_size=18, 
                    font_weight='bold', 
                    edge_color = [self.__colors[randint(0, len(self.__colors)-1)] 
                                  for edge in G.edges()], 
                    width=2.0)
    
    def get_vertex_degrees(self):
        if not self.__is_directed:
            num_vertices = len(self.__adjacency_matrix)
            degrees = []
            
            matrix = np.copy(self.__adjacency_matrix)
            
            for vertex in range(num_vertices):
                matrix[vertex][matrix[vertex] > 0] = 1
                degree = np.sum(matrix[vertex])
                degrees.append(degree)
            print("Vertex degrees: ", degrees)
            return degrees
        else:
            num_vertices = len(self.__adjacency_matrix)
            matrix = np.copy(self.__adjacency_matrix)
            out_degrees = []
            in_degrees = []
            
            for vertex in range(num_vertices):
                matrix[vertex][matrix[vertex] > 0] = 1
#                 print(matrix[vertex])
                out_degrees.append(sum(matrix[vertex]))
                in_degrees.append(sum(row[vertex] 
                                      if row[vertex] < 1 else 1 
                                      for row in matrix))
#             print(matrix)
#             print(self.__is_directed,out_degrees, in_degrees)
            return out_degrees, in_degrees
        
    def floyd_warshall(self):
        num_vertices = len(self.__adjacency_matrix)
        
        
        dist_matrix = np.copy(self.__adjacency_matrix)
        
        dist_matrix[dist_matrix == 0] = 99999999
        np.fill_diagonal(dist_matrix, 0)
#         print(dist_matrix)
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                        dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
        self.__dist_matrix = dist_matrix
        print("S_matrix: ", dist_matrix, sep = "\n")
        print("D:" , self.__diameter(dist_matrix))
        print("R:" , self.__radius(dist_matrix))
        print("Z:" , self.__central_verticals(dist_matrix))
        print("P:", self.__peripheral_vertices(dist_matrix))
        
        
    
    def __radius(self, matrix):
        radius = np.min(np.max(matrix, axis=1))
        return radius
    def __diameter(self, matrix):
        diameter = np.max(matrix)
        return diameter
    def __central_verticals(self, matrix):
        central_vertices = np.where(np.max(matrix, axis=1) == self.__radius(matrix))[0]
        return central_vertices
    def __peripheral_vertices(self, matrix):
        peripheral_vertices = np.where(np.max(matrix, axis=1) == self.__diameter(matrix))[0]
        return peripheral_vertices
    
    
    def output(self, output_file):
        with open(output_file, 'w') as file:
            file.write("Degree Vector: " + str(self.get_vertex_degrees()) + "\n")
            file.write("Distance Matrix:\n" + str(self.floyd_warshall()) + "\n")
            file.write("Diameter: " + str(self.__diameter(self.__dist_matrix)) + "\n")
            file.write("Radius: " + str(self.__radius(self.__dist_matrix)) + "\n")
            file.write("Central Vertices: " + str(self.__central_verticals(self.__dist_matrix)) + "\n")
            file.write("Peripheral Vertices: " + str(self.__peripheral_vertices(self.__dist_matrix)) + "\n")
