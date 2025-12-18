"""
Sorting Algorithms Implementation
Implements various sorting algorithms from scratch
Used for data preprocessing and feature ranking
"""

import numpy as np
import time


class SortingAlgorithms:
    """
    Implementation of various sorting algorithms
    """
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def reset_counters(self):
        """Reset comparison and swap counters"""
        self.comparisons = 0
        self.swaps = 0
    
    def bubble_sort(self, arr):
        """
        Bubble Sort Algorithm
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Used for: Small datasets, educational purposes
        """
        self.reset_counters()
        arr = arr.copy()
        n = len(arr)
        
        print(f"\n=== Bubble Sort ===")
        print(f"Array size: {n}")
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                    swapped = True
            
            if not swapped:
                break
        
        print(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
        return arr
    
    def selection_sort(self, arr):
        """
        Selection Sort Algorithm
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Used for: Small datasets where swaps are expensive
        """
        self.reset_counters()
        arr = arr.copy()
        n = len(arr)
        
        print(f"\n=== Selection Sort ===")
        print(f"Array size: {n}")
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
        
        print(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
        return arr
    
    def insertion_sort(self, arr):
        """
        Insertion Sort Algorithm
        Time Complexity: O(n²) worst case, O(n) best case
        Space Complexity: O(1)
        
        Used for: Nearly sorted data, online sorting
        """
        self.reset_counters()
        arr = arr.copy()
        n = len(arr)
        
        print(f"\n=== Insertion Sort ===")
        print(f"Array size: {n}")
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            
            while j >= 0 and arr[j] > key:
                self.comparisons += 1
                arr[j + 1] = arr[j]
                self.swaps += 1
                j -= 1
            
            arr[j + 1] = key
        
        print(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
        return arr
    
    def merge_sort(self, arr):
        """
        Merge Sort Algorithm
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Used for: Large datasets, stable sorting required
        """
        self.reset_counters()
        
        print(f"\n=== Merge Sort ===")
        print(f"Array size: {len(arr)}")
        
        def merge(left, right):
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                self.comparisons += 1
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        def merge_sort_recursive(arr):
            if len(arr) <= 1:
                return arr
            
            mid = len(arr) // 2
            left = merge_sort_recursive(arr[:mid])
            right = merge_sort_recursive(arr[mid:])
            
            return merge(left, right)
        
        sorted_arr = merge_sort_recursive(arr.tolist() if isinstance(arr, np.ndarray) else arr)
        print(f"Comparisons: {self.comparisons}")
        return np.array(sorted_arr)
    
    def quick_sort(self, arr):
        """
        Quick Sort Algorithm
        Time Complexity: O(n log n) average, O(n²) worst case
        Space Complexity: O(log n)
        
        Used for: General purpose sorting, cache-friendly
        """
        self.reset_counters()
        
        print(f"\n=== Quick Sort ===")
        print(f"Array size: {len(arr)}")
        
        def partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                self.comparisons += 1
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    self.swaps += 1
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            self.swaps += 1
            return i + 1
        
        def quick_sort_recursive(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_recursive(arr, low, pi - 1)
                quick_sort_recursive(arr, pi + 1, high)
        
        arr = arr.copy()
        quick_sort_recursive(arr, 0, len(arr) - 1)
        print(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
        return arr
    
    def heap_sort(self, arr):
        """
        Heap Sort Algorithm
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        
        Used for: When consistent O(n log n) is needed
        """
        self.reset_counters()
        arr = arr.copy()
        n = len(arr)
        
        print(f"\n=== Heap Sort ===")
        print(f"Array size: {n}")
        
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                self.comparisons += 1
                if arr[left] > arr[largest]:
                    largest = left
            
            if right < n:
                self.comparisons += 1
                if arr[right] > arr[largest]:
                    largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.swaps += 1
                heapify(arr, n, largest)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            heapify(arr, i, 0)
        
        print(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
        return arr
    
    def counting_sort(self, arr):
        """
        Counting Sort Algorithm
        Time Complexity: O(n + k) where k is range
        Space Complexity: O(k)
        
        Used for: Integer sorting with small range
        """
        self.reset_counters()
        
        print(f"\n=== Counting Sort ===")
        print(f"Array size: {len(arr)}")
        
        # Convert to integers
        arr_int = arr.astype(int)
        
        max_val = int(np.max(arr_int))
        min_val = int(np.min(arr_int))
        range_val = max_val - min_val + 1
        
        # Create count array
        count = [0] * range_val
        output = [0] * len(arr_int)
        
        # Count occurrences
        for num in arr_int:
            count[num - min_val] += 1
        
        # Cumulative count
        for i in range(1, range_val):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(len(arr_int) - 1, -1, -1):
            output[count[arr_int[i] - min_val] - 1] = arr_int[i]
            count[arr_int[i] - min_val] -= 1
        
        print(f"Range: {range_val}")
        return np.array(output)
    
    def radix_sort(self, arr):
        """
        Radix Sort Algorithm
        Time Complexity: O(d * (n + k))
        Space Complexity: O(n + k)
        
        Used for: Sorting integers, strings
        """
        print(f"\n=== Radix Sort ===")
        print(f"Array size: {len(arr)}")
        
        arr_int = arr.astype(int)
        max_val = int(np.max(arr_int))
        
        exp = 1
        while max_val // exp > 0:
            self._counting_sort_by_digit(arr_int, exp)
            exp *= 10
        
        return arr_int
    
    def _counting_sort_by_digit(self, arr, exp):
        """Helper function for radix sort"""
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        
        for i in range(n):
            arr[i] = output[i]


class SearchAlgorithms:
    """
    Implementation of search algorithms
    """
    
    def linear_search(self, arr, target):
        """
        Linear Search Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        print(f"\n=== Linear Search for {target} ===")
        
        for i, val in enumerate(arr):
            if val == target:
                print(f"Found at index: {i}")
                return i
        
        print("Not found")
        return -1
    
    def binary_search(self, arr, target):
        """
        Binary Search Algorithm
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Note: Requires sorted array
        """
        print(f"\n=== Binary Search for {target} ===")
        
        left, right = 0, len(arr) - 1
        iterations = 0
        
        while left <= right:
            iterations += 1
            mid = (left + right) // 2
            print(f"Iteration {iterations}: Checking index {mid}, value = {arr[mid]}")
            
            if arr[mid] == target:
                print(f"Found at index: {mid}")
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        print("Not found")
        return -1
    
    def binary_search_recursive(self, arr, target, left=0, right=None):
        """
        Recursive Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(log n) due to recursion
        """
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return self.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return self.binary_search_recursive(arr, target, left, mid - 1)


def compare_sorting_algorithms(data):
    """
    Compare performance of different sorting algorithms
    """
    print("\n" + "="*80)
    print("SORTING ALGORITHMS COMPARISON")
    print("="*80)
    
    sorter = SortingAlgorithms()
    algorithms = [
        ('Bubble Sort', sorter.bubble_sort),
        ('Selection Sort', sorter.selection_sort),
        ('Insertion Sort', sorter.insertion_sort),
        ('Merge Sort', sorter.merge_sort),
        ('Quick Sort', sorter.quick_sort),
        ('Heap Sort', sorter.heap_sort),
    ]
    
    results = []
    
    for name, func in algorithms:
        start_time = time.time()
        sorted_data = func(data)
        end_time = time.time()
        
        results.append({
            'algorithm': name,
            'time': end_time - start_time,
            'comparisons': sorter.comparisons,
            'swaps': sorter.swaps
        })
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<20} {'Time (s)':<15} {'Comparisons':<15} {'Swaps':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['algorithm']:<20} {r['time']:<15.6f} {r['comparisons']:<15} {r['swaps']:<15}")
    
    return results


def demonstrate_sorting_algorithms():
    """
    Demonstrate all sorting algorithms
    """
    print("\n" + "="*80)
    print("SORTING & SEARCH ALGORITHMS DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randint(1, 100, size=20)
    print(f"\nOriginal data: {data}")
    
    # Compare all sorting algorithms
    compare_sorting_algorithms(data)
    
    # Demonstrate search algorithms
    searcher = SearchAlgorithms()
    sorted_data = np.sort(data)
    print(f"\nSorted data: {sorted_data}")
    
    target = sorted_data[10]
    searcher.linear_search(data, target)
    searcher.binary_search(sorted_data, target)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demonstrate_sorting_algorithms()
