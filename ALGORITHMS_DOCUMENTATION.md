# Data Structure & Algorithm Concepts Used

## Complete List of Algorithms Implemented

### 1. GRAPH ALGORITHMS

#### **Breadth-First Search (BFS)**
- **File**: `graph_algorithms.py` (Lines 35-61)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Usage**: Finding shortest path in patient similarity networks, level-order traversal
- **Application**: Identifies patients with similar health profiles within N-hops

#### **Depth-First Search (DFS)**
- **File**: `graph_algorithms.py` (Lines 63-83)
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Usage**: Detecting connected components, exploring feature dependency trees
- **Application**: Groups patients into clusters based on similarity

#### **Dijkstra's Shortest Path Algorithm**
- **File**: `graph_algorithms.py` (Lines 85-118)
- **Time Complexity**: O((V + E) log V) with min-heap
- **Space Complexity**: O(V)
- **Usage**: Finding optimal treatment paths, computing patient risk progression
- **Application**: Determines minimum distance between patients in similarity network

#### **Bellman-Ford Algorithm**
- **File**: `graph_algorithms.py` (Lines 128-165)
- **Time Complexity**: O(V × E)
- **Space Complexity**: O(V)
- **Usage**: Handles negative weights, detects negative cycles
- **Application**: Validates graph consistency in patient progression models

#### **Kruskal's Minimum Spanning Tree**
- **File**: `graph_algorithms.py` (Lines 167-219)
- **Time Complexity**: O(E log E)
- **Space Complexity**: O(V)
- **Algorithm Components**:
  - Edge sorting
  - Union-Find with path compression
  - Union by rank optimization
- **Usage**: Finding minimum feature correlation network
- **Application**: Identifies most important feature relationships

#### **Prim's Minimum Spanning Tree**
- **File**: `graph_algorithms.py` (Lines 221-259)
- **Time Complexity**: O(E log V) with min-heap
- **Space Complexity**: O(V)
- **Usage**: Alternative MST construction for feature networks
- **Application**: Builds optimal feature dependency tree

### 2. SORTING ALGORITHMS

#### **Bubble Sort**
- **File**: `sorting_algorithms.py` (Lines 22-46)
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: Yes
- **Usage**: Educational purposes, small datasets
- **Optimization**: Early termination when no swaps occur

#### **Selection Sort**
- **File**: `sorting_algorithms.py` (Lines 48-73)
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: No
- **Usage**: Small datasets where swaps are expensive
- **Application**: Sorting feature importance scores

#### **Insertion Sort**
- **File**: `sorting_algorithms.py` (Lines 75-102)
- **Time Complexity**: O(n²) worst, O(n) best
- **Space Complexity**: O(1)
- **Stable**: Yes
- **Usage**: Nearly sorted data, online sorting
- **Application**: Maintaining sorted patient lists

#### **Merge Sort**
- **File**: `sorting_algorithms.py` (Lines 104-140)
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(n)
- **Stable**: Yes
- **Algorithm**: Divide and conquer with merging
- **Usage**: Large datasets, guaranteed O(n log n)
- **Application**: Sorting feature variances for analysis

#### **Quick Sort**
- **File**: `sorting_algorithms.py` (Lines 142-178)
- **Time Complexity**: O(n log n) average, O(n²) worst
- **Space Complexity**: O(log n)
- **Stable**: No
- **Algorithm**: Partitioning with pivot selection
- **Usage**: General-purpose sorting, cache-friendly
- **Application**: Default sorting for feature ranking

#### **Heap Sort**
- **File**: `sorting_algorithms.py` (Lines 180-223)
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(1)
- **Stable**: No
- **Algorithm**: Max-heap construction and extraction
- **Usage**: Consistent O(n log n) performance needed
- **Application**: Priority-based feature selection

#### **Counting Sort**
- **File**: `sorting_algorithms.py` (Lines 225-257)
- **Time Complexity**: O(n + k) where k is range
- **Space Complexity**: O(k)
- **Stable**: Yes
- **Usage**: Integer sorting with small range
- **Application**: Sorting patient age groups, categorical features

#### **Radix Sort**
- **File**: `sorting_algorithms.py` (Lines 259-279)
- **Time Complexity**: O(d × (n + k))
- **Space Complexity**: O(n + k)
- **Algorithm**: Digit-by-digit sorting
- **Usage**: Sorting integers, multi-key sorting
- **Application**: Organizing patient IDs, medical record numbers

### 3. SEARCHING ALGORITHMS

#### **Linear Search**
- **File**: `sorting_algorithms.py` (Lines 290-302)
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Usage**: Unsorted data, small datasets
- **Application**: Finding patients by exact criteria

#### **Binary Search (Iterative)**
- **File**: `sorting_algorithms.py` (Lines 304-326)
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1)
- **Requirement**: Sorted array
- **Usage**: Fast searching in sorted data
- **Application**: Finding features within variance threshold

#### **Binary Search (Recursive)**
- **File**: `sorting_algorithms.py` (Lines 328-346)
- **Time Complexity**: O(log n)
- **Space Complexity**: O(log n) due to recursion
- **Usage**: Recursive implementation alternative
- **Application**: Feature lookup in sorted arrays

### 4. TREE ALGORITHMS

#### **Heap Operations (Heapify)**
- **File**: `sorting_algorithms.py` (Lines 195-210)
- **Time Complexity**: O(log n) per operation
- **Usage**: Priority queue operations in Dijkstra and Prim's
- **Application**: Managing priority in graph algorithms

#### **Union-Find (Disjoint Set)**
- **File**: `graph_algorithms.py` (Lines 195-209)
- **Operations**:
  - Find with path compression: O(α(n)) ≈ O(1)
  - Union by rank: O(α(n)) ≈ O(1)
- **Usage**: Detecting cycles in Kruskal's algorithm
- **Application**: Merging patient clusters

### 5. DATA STRUCTURE CONCEPTS

#### **Graph Representations**
- **Adjacency List**: `graph_algorithms.py` (Line 21)
  - Space: O(V + E)
  - Good for sparse graphs
- **Adjacency Matrix**: `graph_algorithms.py` (Lines 22-24)
  - Space: O(V²)
  - Fast edge lookup: O(1)

#### **Priority Queue (Min-Heap)**
- **Implementation**: Python's heapq module
- **Usage**: Dijkstra, Prim's algorithms
- **Operations**: Insert O(log n), Extract-min O(log n)

#### **Queue (FIFO)**
- **Implementation**: collections.deque
- **Usage**: BFS algorithm
- **Operations**: Enqueue O(1), Dequeue O(1)

#### **Stack (LIFO)**
- **Implementation**: Recursion call stack
- **Usage**: DFS algorithm
- **Operations**: Push O(1), Pop O(1)

### 6. ALGORITHM DESIGN PARADIGMS

#### **Divide and Conquer**
- **Examples**: Merge Sort, Quick Sort
- **Pattern**: Divide problem, solve subproblems, combine solutions
- **Application**: Efficient sorting of large datasets

#### **Greedy Algorithms**
- **Examples**: Dijkstra, Prim's, Kruskal's
- **Pattern**: Make locally optimal choice at each step
- **Application**: Finding MST in feature correlation networks

#### **Dynamic Programming**
- **Example**: Bellman-Ford algorithm
- **Pattern**: Store solutions to overlapping subproblems
- **Application**: Shortest path with negative weights

#### **Backtracking**
- **Example**: DFS with path tracking
- **Pattern**: Explore all possibilities, backtrack on failure
- **Application**: Finding all paths in patient networks

### 7. OPTIMIZATION TECHNIQUES

#### **Path Compression** (Union-Find)
- Flattens tree structure during find operations
- Amortized time: O(α(n))

#### **Union by Rank** (Union-Find)
- Keeps tree balanced by attaching smaller tree to larger
- Prevents degeneration to linked list

#### **Early Termination** (Bubble Sort)
- Stops if no swaps occur in a pass
- Best case improves to O(n)

#### **Memoization** (DFS)
- Caches visited nodes
- Prevents revisiting nodes

### 8. COMPLEXITY ANALYSIS SUMMARY

| Algorithm | Best Case | Average Case | Worst Case | Space |
|-----------|-----------|--------------|------------|-------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) |
| Radix Sort | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) |
| BFS | O(V+E) | O(V+E) | O(V+E) | O(V) |
| DFS | O(V+E) | O(V+E) | O(V+E) | O(V) |
| Dijkstra | O((V+E)logV) | O((V+E)logV) | O((V+E)logV) | O(V) |
| Bellman-Ford | O(VE) | O(VE) | O(VE) | O(V) |
| Kruskal | O(E log E) | O(E log E) | O(E log E) | O(V) |
| Prim | O((V+E)logV) | O((V+E)logV) | O((V+E)logV) | O(V) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Linear Search | O(1) | O(n) | O(n) | O(1) |

### 9. PRACTICAL APPLICATIONS IN PROJECT

1. **Patient Similarity Network** (BFS, DFS, Dijkstra)
   - Builds graph of similar patients
   - Finds shortest path between patients
   - Identifies patient clusters

2. **Feature Correlation Analysis** (Kruskal, Prim)
   - Creates MST of highly correlated features
   - Reduces redundant features
   - Identifies core feature relationships

3. **Data Preprocessing** (All Sorting Algorithms)
   - Ranks features by importance
   - Sorts patients by risk score
   - Orders data for binary search

4. **Efficient Searching** (Binary Search)
   - Quick feature lookup
   - Finding patients in sorted lists
   - Threshold-based filtering

### 10. CODE INSTRUMENTATION

Each algorithm includes:
- **Comparison counter**: Tracks number of comparisons
- **Swap counter**: Counts element movements
- **Step-by-step output**: Shows algorithm progress
- **Performance metrics**: Measures execution time

### 11. TESTING & DEMONSTRATION

Run individual algorithm modules:
```bash
# Test graph algorithms
python graph_algorithms.py

# Test sorting algorithms
python sorting_algorithms.py

# Full training with all algorithms
python train_models.py
```

All algorithms are integrated into the main training pipeline and execute automatically!
