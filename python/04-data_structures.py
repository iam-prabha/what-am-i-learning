# Converted from 04-data_structures.ipynb

# ======================================================================
# # Python Built-in Data Structures with Methods
# 
# This notebook covers Python's built-in data structures: **List**, **Set**, **Tuple**, and **Dictionary**, including their key methods and common techniques.
# ======================================================================

# ======================================================================
# ## 1. List
# 
# - **Definition**: Ordered, mutable collection of items.
# - **Use case**: Storing sequences of items that may change.
# - **Syntax**: `my_list = [item1, item2, ...]`
# ======================================================================

# %%
# Creating a list
fruits = ["apple", "mango", "orange"]
print("List:", fruits)

# %%
# Key Methods
# 1. append(item): Add item to end
fruits.append("grape")
print("After append:", fruits)

# %%
# 2. insert(index, item): Insert item at index
fruits.insert(1, "mango")
print("After insert:", fruits)

# %%
# 3. remove(item): Remove first occurrence of item
fruits.remove("mango")
print("After remove:", fruits)

# %%
# 4. pop(index): Remove and return item at index (default: last)
popped = fruits.pop()
print("Popped item:", popped, "| List:", fruits)

# %%
# 5. sort(): Sort list in place
fruits.sort()
print("Sorted:", fruits)

# %%
# Technique: List comprehension
squares = [x**2 for x in range(5)]
print("Squares (comprehension):", squares)

# ======================================================================
# ## 2. Set
# 
# - **Definition**: Unordered, mutable collection of unique items.
# - **Use case**: Removing duplicates or performing set operations.
# - **Syntax**: `my_set = {item1, item2, ...}`
# ======================================================================

# %%
# Creating a set
numbers = {1, 2, 2, 3, 4}
print("Set:", numbers)  # Duplicates removed

# %%
# Key Methods
# 1. add(item): Add item to set
numbers.add(5)
print("After add:", numbers)

# %%
# 2. remove(item): Remove item (raises KeyError if not found)
numbers.remove(3)
print("After remove:", numbers)

# %%
# 3. discard(item): Remove item (no error if not found)
numbers.discard(10)  # No error
print("After discard:", numbers)

# %%
# 4. union(other_set): Return union of sets
other_set = {4, 5, 6}
union = numbers.union(other_set)
print("Union:", union)

# %%
# 5. intersection(other_set): Return intersection of sets
intersection = numbers.intersection(other_set)
print("Intersection:", intersection)

# %%
# Technique: Set comprehension
evens = {x for x in range(10) if x % 2 == 0}
print("Evens (comprehension):", evens)

# ======================================================================
# ## 3. Tuple
# 
# - **Definition**: Ordered, immutable collection of items.
# - **Use case**: Storing fixed data that shouldn't change.
# - **Syntax**: `my_tuple = (item1, item2, ...)`
# ======================================================================

# %%
# Creating a tuple
coords = (10, 20, 30)
print("Tuple:", coords)

# %%
# Key Methods
# 1. count(item): Count occurrences of item
values = (1, 2, 2, 3)
count = values.count(2)
print("Count of 2:", count)

# %%
# 2. index(item): Return first index of item
index = values.index(3)
print("Index of 3:", index)

# %%
# Technique: Unpacking
x, y, z = coords
print("Unpacked:", x, y, z)

# ======================================================================
# ## 4. Dictionary
# 
# - **Definition**: Unordered, mutable collection of key-value pairs.
# - **Use case**: Storing data with unique keys for quick lookup.
# - **Syntax**: `my_dict = {key1: value1, key2: value2, ...}`
# ======================================================================

# %%
# Creating a dictionary
student = {"name": "Alice", "age": 20, "grade": "A"}
print("Dictionary:", student)

# %%
# Key Methods
# 1. get(key, default): Get value for key, return default if not found
score = student.get("score", "N/A")
print("Score (default):", score)

# %%
# 2. update(other_dict): Update dictionary with key-value pairs
student.update({"age": 21, "score": 95})
print("After update:", student)

# %%
# 3. pop(key): Remove and return value for key
grade = student.pop("grade")
print("Popped grade:", grade, "| Dictionary:", student)

# %%
# 4. keys(): Return view of dictionary keys
keys = student.keys()
print("Keys:", keys)

# %%
# 5. values(): Return view of dictionary values
values = student.values()
print("Values:", values)

# %%
# Technique: Dictionary comprehension
squared_dict = {x: x**2 for x in range(4)}
print("Squared dict (comprehension):", squared_dict)

# %%


