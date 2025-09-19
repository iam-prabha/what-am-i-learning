# static array , Dynamic array and string

a = [1, 2, 3, 4, 5]

# append - insert at the end - on average O(1)
a.append(6)

# pop - remove at the end - on average O(1)
a.pop()

# insert ( not at the end ) - on average O(n)
a.insert(2, 8)

# modify - on average O(1)
a[0] = 2

# access - on average O(1)
print(a[0])

# search - on average O(n)
print(a.index(2))

# delete - on average O(n)
a.remove(2)
print(a)

# size - on average O(1)
print(len(a))

# Strings

# Append to end of string - O(n)
s = 'hello'

b = s + 'z'

# Check if something is in string - O(n)
if 'f' in s:
  print(True)

# Access positions - O(1)
print(s[2])

# Check length of string - O(1)
print(len(s))