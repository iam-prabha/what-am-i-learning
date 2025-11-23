# Data Structures & Algorithms

Implementation of fundamental data structures and algorithms in Python.

## 📁 Structure

### Data Structures
Common data structures implemented from scratch:

- **data_structures/00_arrays_and_strings.py** - Array operations, string manipulation
- **data_structures/01_linked_list.py** - Singly and doubly linked lists
- **data_structures/02_stacks.py** - Stack implementation and applications
- **data_structures/03_queue.py** - Queue, circular queue, deque
- **data_structures/04_hash_tables.py** - Hash maps, collision handling
- **data_structures/05_heap.py** - Min heap, max heap, priority queue
- **data_structures/reverse_str.py** - String reversal examples and techniques

### Algorithms
Classic algorithms organized by category (implementations coming soon):

- **algorithms/sorting/** - Sorting algorithms (bubble, merge, quick, heap sort)
- **algorithms/searching/** - Search algorithms (binary search, linear search)
- **algorithms/graph/** - Graph algorithms (BFS, DFS, shortest path)

## 🎯 Learning Path

1. Master basic data structures (arrays, linked lists, stacks, queues)
2. Understand hash tables and their applications
3. Learn tree and heap structures
4. Practice sorting algorithms
5. Explore graph algorithms

## 💡 Practice Tips

- Implement each data structure from scratch before using built-ins
- Analyze time and space complexity
- Solve problems on LeetCode, HackerRank, or CodeForces
- Focus on understanding, not memorization

## 📊 Complexity Reference

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array         | O(1)   | O(n)   | O(n)   | O(n)   |
| Linked List   | O(n)   | O(n)   | O(1)   | O(1)   |
| Stack         | O(n)   | O(n)   | O(1)   | O(1)   |
| Queue         | O(n)   | O(n)   | O(1)   | O(1)   |
| Hash Table    | N/A    | O(1)   | O(1)   | O(1)   |
| Binary Heap   | N/A    | O(n)   | O(log n) | O(log n) |

## 💻 Usage

### Running Python Scripts
```bash
# From repository root
uv run python dsa/data_structures/00_arrays_and_strings.py
uv run python dsa/data_structures/01_linked_list.py
# ... and so on
```

### Testing Implementations
```bash
# Run individual data structure tests
uv run python -m pytest dsa/data_structures/ -v
```

## 🔗 Resources

- Python fundamentals: `../python/`
- Algorithm theory: See `algorithms/README.md` for detailed algorithm documentation
