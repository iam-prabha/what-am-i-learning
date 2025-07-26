
"""
Arrays & Strings - Data Structures & Algorithms Course.ipynb
"""

# Arrays

# Access: O(1)
a = [1,2,3,4,5]
print(a[3])

# Append: O(1)
a.append(6)
print(a)

# Insert/delete: O(n)
# insert at middle
a.insert(2, 'hello')
print(a)

# delete at anywhere & end values(pop)
a.remove(2)
print(a)
a.pop()
print(a)


# Strings

# chars = 'hello'