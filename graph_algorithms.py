"""
Graph Algorithms Implementation
Implements BFS, DFS, Dijkstra, Bellman-Ford, Kruskal, Prim's MST
Used for patient similarity networks and feature correlation analysis
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict
import heapq


class Graph:
    """Graph data structure with various algorithms"""
    
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
        self.adj_matrix = [[float('inf')] * vertices for _ in range(vertices)]
        for i in range(vertices):
            self.adj_matrix[i][i] = 0
    
    def add_edge(self, u, v, weight=1):
        """Add weighted edge to graph"""
        self.graph[u].append((v, weight))
        self.adj_matrix[u][v] = weight
    
    def add_undirected_edge(self, u, v, weight=1):
        """Add undirected weighted edge"""
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
        self.adj_matrix[u][v] = weight
        self.adj_matrix[v][u] = weight
    
    def bfs(self, start):
        """
        Breadth-First Search Algorithm
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Used for: Finding shortest path in unweighted graphs,
                  Level-order traversal of patient similarity networks
        """
        visited = [False] * self.V
        queue = deque([start])
        visited[start] = True
        traversal_order = []
        distances = [-1] * self.V
        distances[start] = 0
        
        print(f"\n=== BFS Traversal from node {start} ===")
        
        while queue:
            node = queue.popleft()
            traversal_order.append(node)
            print(f"Visiting node: {node} (Distance: {distances[node]})")
            
            for neighbor, _ in self.graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        return traversal_order, distances
    
    def dfs(self, start):
        """
        Depth-First Search Algorithm
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Used for: Detecting cycles, pathfinding,
                  Exploring feature dependency trees
        """
        visited = [False] * self.V
        traversal_order = []
        
        print(f"\n=== DFS Traversal from node {start} ===")
        
        def dfs_recursive(node):
            visited[node] = True
            traversal_order.append(node)
            print(f"Visiting node: {node}")
            
            for neighbor, _ in self.graph[node]:
                if not visited[neighbor]:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return traversal_order
    
    def dijkstra(self, start):
        """
        Dijkstra's Shortest Path Algorithm
        Time Complexity: O((V + E) log V) with min-heap
        Space Complexity: O(V)
        
        Used for: Finding optimal treatment paths,
                  Computing patient risk progression paths
        """
        distances = [float('inf')] * self.V
        distances[start] = 0
        parent = [-1] * self.V
        min_heap = [(0, start)]
        visited = [False] * self.V
        
        print(f"\n=== Dijkstra's Algorithm from node {start} ===")
        
        while min_heap:
            dist, u = heapq.heappop(min_heap)
            
            if visited[u]:
                continue
            
            visited[u] = True
            print(f"Processing node {u}, Distance: {dist}")
            
            for v, weight in self.graph[u]:
                if not visited[v] and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    parent[v] = u
                    heapq.heappush(min_heap, (distances[v], v))
        
        return distances, parent
    
    def get_shortest_path(self, parent, start, end):
        """Reconstruct shortest path from Dijkstra's parent array"""
        path = []
        current = end
        
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        
        if path[0] == start:
            return path
        return []
    
    def bellman_ford(self, start):
        """
        Bellman-Ford Algorithm
        Time Complexity: O(V * E)
        Space Complexity: O(V)
        
        Used for: Handling negative weights in patient progression,
                  Detecting negative cycles in treatment effectiveness
        """
        distances = [float('inf')] * self.V
        distances[start] = 0
        parent = [-1] * self.V
        
        print(f"\n=== Bellman-Ford Algorithm from node {start} ===")
        
        # Relax all edges V-1 times
        for i in range(self.V - 1):
            for u in range(self.V):
                for v, weight in self.graph[u]:
                    if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        parent[v] = u
                        print(f"Iteration {i+1}: Updated distance to {v} = {distances[v]}")
        
        # Check for negative cycles
        has_negative_cycle = False
        for u in range(self.V):
            for v, weight in self.graph[u]:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    has_negative_cycle = True
                    print("WARNING: Negative cycle detected!")
                    break
        
        return distances, parent, not has_negative_cycle
    
    def kruskal_mst(self):
        """
        Kruskal's Minimum Spanning Tree Algorithm
        Time Complexity: O(E log E)
        Space Complexity: O(V)
        
        Used for: Finding minimum feature correlation network,
                  Building optimal patient similarity trees
        """
        print("\n=== Kruskal's MST Algorithm ===")
        
        # Collect all edges
        edges = []
        for u in range(self.V):
            for v, weight in self.graph[u]:
                if u < v:  # Avoid duplicate edges in undirected graph
                    edges.append((weight, u, v))
        
        # Sort edges by weight
        edges.sort()
        
        # Union-Find data structure
        parent = list(range(self.V))
        rank = [0] * self.V
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        mst_edges = []
        mst_weight = 0
        
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                mst_weight += weight
                print(f"Added edge ({u}, {v}) with weight {weight:.4f}")
                
                if len(mst_edges) == self.V - 1:
                    break
        
        print(f"Total MST weight: {mst_weight:.4f}")
        return mst_edges, mst_weight
    
    def prim_mst(self, start=0):
        """
        Prim's Minimum Spanning Tree Algorithm
        Time Complexity: O(E log V) with min-heap
        Space Complexity: O(V)
        
        Used for: Alternative MST construction for feature networks
        """
        print(f"\n=== Prim's MST Algorithm starting from {start} ===")
        
        in_mst = [False] * self.V
        key = [float('inf')] * self.V
        parent = [-1] * self.V
        min_heap = [(0, start)]
        key[start] = 0
        
        mst_edges = []
        mst_weight = 0
        
        while min_heap:
            weight, u = heapq.heappop(min_heap)
            
            if in_mst[u]:
                continue
            
            in_mst[u] = True
            
            if parent[u] != -1:
                mst_edges.append((parent[u], u, weight))
                mst_weight += weight
                print(f"Added edge ({parent[u]}, {u}) with weight {weight:.4f}")
            
            for v, w in self.graph[u]:
                if not in_mst[v] and w < key[v]:
                    key[v] = w
                    parent[v] = u
                    heapq.heappush(min_heap, (w, v))
        
        print(f"Total MST weight: {mst_weight:.4f}")
        return mst_edges, mst_weight


class PatientSimilarityGraph:
    """
    Build patient similarity network using graph algorithms
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_patients = len(X)
        self.graph = Graph(self.n_patients)
    
    def build_similarity_graph(self, threshold=0.7):
        """
        Build graph where edges represent patient similarity
        Uses Euclidean distance as similarity metric
        """
        print(f"\n=== Building Patient Similarity Graph ===")
        print(f"Number of patients: {self.n_patients}")
        print(f"Similarity threshold: {threshold}")
        
        edge_count = 0
        
        for i in range(self.n_patients):
            for j in range(i + 1, self.n_patients):
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2))
                
                # Convert distance to similarity (inverse relationship)
                max_dist = np.sqrt(self.X.shape[1])  # Maximum possible distance
                similarity = 1 - (distance / max_dist)
                
                # Add edge if similarity exceeds threshold
                if similarity >= threshold:
                    self.graph.add_undirected_edge(i, j, 1 - similarity)
                    edge_count += 1
        
        print(f"Created {edge_count} edges in similarity graph")
        return self.graph
    
    def find_similar_patients_bfs(self, patient_id, max_distance=2):
        """
        Use BFS to find similar patients within max_distance hops
        """
        _, distances = self.graph.bfs(patient_id)
        
        similar_patients = []
        for i, dist in enumerate(distances):
            if 0 < dist <= max_distance:
                similar_patients.append((i, dist))
        
        return similar_patients
    
    def find_patient_clusters_dfs(self):
        """
        Use DFS to identify patient clusters (connected components)
        """
        visited = [False] * self.n_patients
        clusters = []
        
        for i in range(self.n_patients):
            if not visited[i]:
                cluster = []
                
                def dfs_cluster(node):
                    visited[node] = True
                    cluster.append(node)
                    for neighbor, _ in self.graph.graph[node]:
                        if not visited[neighbor]:
                            dfs_cluster(neighbor)
                
                dfs_cluster(i)
                clusters.append(cluster)
        
        return clusters


class FeatureCorrelationGraph:
    """
    Build feature correlation network using graph algorithms
    """
    
    def __init__(self, X, feature_names):
        self.X = X
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.graph = Graph(self.n_features)
    
    def build_correlation_graph(self, threshold=0.5):
        """
        Build graph where edges represent feature correlations
        """
        print(f"\n=== Building Feature Correlation Graph ===")
        print(f"Number of features: {self.n_features}")
        print(f"Correlation threshold: {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.X.T)
        
        edge_count = 0
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                correlation = abs(corr_matrix[i, j])
                
                if correlation >= threshold:
                    # Use inverse correlation as weight (lower weight = higher correlation)
                    weight = 1 - correlation
                    self.graph.add_undirected_edge(i, j, weight)
                    edge_count += 1
                    print(f"Feature {self.feature_names[i]} <-> {self.feature_names[j]}: {correlation:.4f}")
        
        print(f"Created {edge_count} edges in correlation graph")
        return self.graph
    
    def find_feature_mst_kruskal(self):
        """
        Find minimum spanning tree of features using Kruskal's algorithm
        Returns most important feature relationships
        """
        mst_edges, mst_weight = self.graph.kruskal_mst()
        
        # Convert to feature names
        feature_relationships = []
        for u, v, weight in mst_edges:
            feature_relationships.append({
                'feature1': self.feature_names[u],
                'feature2': self.feature_names[v],
                'correlation': 1 - weight,
                'weight': weight
            })
        
        return feature_relationships
    
    def find_feature_mst_prim(self):
        """
        Find minimum spanning tree using Prim's algorithm
        """
        mst_edges, mst_weight = self.graph.prim_mst()
        
        feature_relationships = []
        for u, v, weight in mst_edges:
            feature_relationships.append({
                'feature1': self.feature_names[u],
                'feature2': self.feature_names[v],
                'correlation': 1 - weight,
                'weight': weight
            })
        
        return feature_relationships


def demonstrate_graph_algorithms():
    """
    Demonstrate all graph algorithms with a sample graph
    """
    print("\n" + "="*80)
    print("GRAPH ALGORITHMS DEMONSTRATION")
    print("="*80)
    
    # Create sample graph
    g = Graph(6)
    g.add_undirected_edge(0, 1, 4)
    g.add_undirected_edge(0, 2, 3)
    g.add_undirected_edge(1, 2, 1)
    g.add_undirected_edge(1, 3, 2)
    g.add_undirected_edge(2, 3, 4)
    g.add_undirected_edge(3, 4, 2)
    g.add_undirected_edge(4, 5, 6)
    
    # Run BFS
    traversal, distances = g.bfs(0)
    print(f"BFS Result: {traversal}")
    
    # Run DFS
    traversal = g.dfs(0)
    print(f"DFS Result: {traversal}")
    
    # Run Dijkstra
    distances, parent = g.dijkstra(0)
    print(f"Dijkstra distances from node 0: {distances}")
    path = g.get_shortest_path(parent, 0, 5)
    print(f"Shortest path from 0 to 5: {path}")
    
    # Run Bellman-Ford
    distances, parent, valid = g.bellman_ford(0)
    print(f"Bellman-Ford distances: {distances}")
    print(f"Graph is valid (no negative cycles): {valid}")
    
    # Run Kruskal's MST
    mst_edges, mst_weight = g.kruskal_mst()
    
    # Run Prim's MST
    mst_edges, mst_weight = g.prim_mst()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate_graph_algorithms()
