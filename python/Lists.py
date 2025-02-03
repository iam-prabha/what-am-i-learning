# algorithm
#  -set of rules or steps used to solve problem

# data structures
# -way of organizing data in a computer

# List is mutable 
# string is immutable trigger err like typeErr

# list methods
# append() => concat or add new element in list
# pop() => remove last element in list or remove element by index like pop(1).
# remove() => remove element by value like remove('banana') &remove only first occurance
# insert() => insert element by index like insert(1, 'kiwi')
# extennd() => add list to another list like list1.extend(list2)
# sort() => sorted by in order
# built-in function and list
# len() => returns total length start in 1...n-1
# max(value) => find max value in list
# min(value) => find min value in list
# sum(value) => sum all from the Firstindex to lastIndex

# SAMPLE CODE
# lst = ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango']
# index : 0,        1,      2,           3,     4,      5,        6
# index : -7,      -6,     -5,         -4,    -3,     -2,      -1 - negative index eg: -1 is mango.-2 is melon

# NESTED LIST SAMPLE CODE
# lst = [
#     [1, 2, 3],
#     [4, 5, 6]
# ]
# print(lst[0][1][0])

#defalut list parameter not recommended way to use
# def func(lst = []):
#     lst.append(1)
#     print(lst)

# silicing list
# lst [1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(lst[2:5]) # [3, 4, 5] not include 5 index with single colon
# print(lst[1:5:2]) # start, end, step [2, 4] with double colon but end index not include
# print(lst[::-1]) # reverse list [9, 8, 7, 6, 5, 4, 3, 2, 1]

# zip() => combine two list of create as tuple
# names = ['apple', 'banana', 'cherry']
# ages = [1, 2, 3]
# zipped = zip(names, ages)
# print(list(zipped)) # [('apple', 1), ('banana', 2), ('cherry', 3)]

# List Comprehension
# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# lst = [x for x in range(10) if x % 2 == 0] => [0, 2, 4, 6, 8]
# lst = [[x for x in range(3)] for _ in range(3)] => [[0, 1, 2], [0, 1, 2], [0, 1, 2]].
# print(lst)