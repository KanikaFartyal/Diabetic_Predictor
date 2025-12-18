# 🎯 Data Structures & Algorithms - Quick Reference

## ✅ All Algorithms Implemented & Demonstrated

### 📊 **GRAPH ALGORITHMS** (graph_algorithms.py)

| Algorithm | Complexity | Line # | Application |
|-----------|-----------|--------|-------------|
| **BFS** | O(V+E) | 35-61 | Patient similarity search, shortest unweighted path |
| **DFS** | O(V+E) | 63-83 | Patient clustering, connected components |
| **Dijkstra** | O((V+E)logV) | 85-118 | Shortest weighted path in similarity network |
| **Bellman-Ford** | O(VE) | 128-165 | Negative cycle detection, all-pairs shortest path |
| **Kruskal's MST** | O(E log E) | 167-219 | Feature correlation network (greedy + Union-Find) |
| **Prim's MST** | O((V+E)logV) | 221-259 | Alternative MST for feature relationships |

**Union-Find Data Structure:** Path compression + Union by rank (Lines 195-209)

### 🔢 **SORTING ALGORITHMS** (sorting_algorithms.py)

| Algorithm | Best | Average | Worst | Stable | Line # |
|-----------|------|---------|-------|--------|--------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | ✅ | 22-46 |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | ❌ | 48-73 |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | ✅ | 75-102 |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | ✅ | 104-140 |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | ❌ | 142-178 |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | ❌ | 180-223 |
| **Counting Sort** | O(n+k) | O(n+k) | O(n+k) | ✅ | 225-257 |
| **Radix Sort** | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | ✅ | 259-279 |

### 🔍 **SEARCHING ALGORITHMS** (sorting_algorithms.py)

| Algorithm | Complexity | Requirements | Line # |
|-----------|-----------|--------------|--------|
| **Linear Search** | O(n) | None | 290-302 |
| **Binary Search** | O(log n) | Sorted array | 304-326 |
| **Binary Search (Recursive)** | O(log n) | Sorted array | 328-346 |

### 🌲 **TREE & HEAP ALGORITHMS**

| Operation | Complexity | Usage |
|-----------|-----------|-------|
| **Heapify** | O(log n) | Building max/min heap (Line 195-210) |
| **Heap Extract** | O(log n) | Priority queue operations |
| **Heap Insert** | O(log n) | Dijkstra, Prim's algorithms |

### 📈 **WHERE ALGORITHMS ARE USED**

#### **In train_models.py:**

```
Line 350: Quick Sort - Sorting feature variances
Line 356: Algorithm Comparison - All 6 sorting algorithms on features
Line 365: Binary Search - Finding features by variance threshold
Line 369: BFS - Finding similar patients within N hops
Line 375: DFS - Identifying patient clusters
Line 381: Dijkstra - Shortest path between patients
Line 392: Kruskal - MST of feature correlations
Line 399: Prim - Alternative MST construction
Line 405: Bellman-Ford - Negative cycle detection
```

#### **Sample Output:**
```
SORTING ALGORITHMS COMPARISON
Algorithm            Time (s)        Comparisons     Swaps
Bubble Sort          0.000036        44              30
Selection Sort       0.000024        45              8
Insertion Sort       0.000022        30              30
Merge Sort           0.000034        23              0
Quick Sort           0.000023        23              19
Heap Sort            0.000083        39              26

KRUSKAL'S MST: Most Important Feature Relationships
  Glucose <-> BloodPressure: correlation = 0.4665
  BloodPressure <-> DiabetesPedigreeFunction: correlation = 0.4203
  Glucose <-> Age: correlation = 0.4186

DIJKSTRA: Shortest Path Between Patients
Distances from Patient 0: [0, 1.2263, 1.3002, 1.9865...]
```

### 🎓 **ALGORITHM DESIGN PATTERNS USED**

1. **Divide & Conquer**: Merge Sort, Quick Sort
2. **Greedy**: Dijkstra, Kruskal, Prim
3. **Dynamic Programming**: Bellman-Ford
4. **Backtracking**: DFS with path tracking
5. **Two Pointers**: Merge in Merge Sort
6. **Sliding Window**: Not directly used
7. **Union-Find**: Kruskal's MST

### 🚀 **HOW TO RUN**

**Test individual algorithms:**
```bash
# Graph algorithms demo
python graph_algorithms.py

# Sorting algorithms demo
python sorting_algorithms.py
```

**Full integration (recommended):**
```bash
# Trains ML models + demonstrates all algorithms
python train_models.py
```

### 📊 **REAL-WORLD APPLICATIONS**

| Algorithm | Healthcare Application |
|-----------|----------------------|
| BFS | Find patients with similar symptoms |
| DFS | Identify disease clusters |
| Dijkstra | Optimal treatment path finding |
| Kruskal/Prim | Discover core feature relationships |
| Quick Sort | Rank patients by risk score |
| Binary Search | Fast patient lookup in sorted records |
| Heap | Priority-based patient triage |

### 🧪 **ALGORITHM ANALYSIS**

Each algorithm tracks:
- ✅ **Comparisons made**
- ✅ **Swaps/moves performed**
- ✅ **Execution time**
- ✅ **Space complexity**
- ✅ **Step-by-step visualization**

### 📚 **KEY DATA STRUCTURES**

| Structure | Implementation | Used In |
|-----------|---------------|---------|
| **Graph (Adjacency List)** | defaultdict(list) | BFS, DFS, Dijkstra |
| **Graph (Adjacency Matrix)** | 2D array | Dense graphs |
| **Min-Heap** | heapq | Dijkstra, Prim |
| **Queue (FIFO)** | deque | BFS |
| **Stack (LIFO)** | Recursion stack | DFS |
| **Union-Find** | Parent array + rank | Kruskal |
| **Priority Queue** | heapq | Dijkstra, Prim |

### 🎯 **COMPLEXITY SUMMARY**

**Graph Operations:**
- Add edge: O(1)
- BFS/DFS: O(V+E)
- Dijkstra: O((V+E) log V) with heap
- Kruskal: O(E log E) for sorting edges
- Prim: O((V+E) log V) with heap

**Sorting:**
- Best general: Merge Sort O(n log n) guaranteed
- Best average: Quick Sort O(n log n)
- Best for small data: Insertion Sort
- Best for integers: Counting/Radix Sort

### ✨ **UNIQUE FEATURES**

- ✅ **Union-Find with path compression** (amortized O(α(n)))
- ✅ **Greedy algorithms with proof of correctness**
- ✅ **Multiple MST algorithms for comparison**
- ✅ **Instrumented code with performance counters**
- ✅ **Real healthcare data applications**
- ✅ **Complete graph traversal implementations**

All algorithms are production-ready with proper error handling and documentation! 🎉
