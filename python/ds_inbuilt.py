# List - ordered, mutable, allows duplicate elements
roles = ['admin', 'user', 'editor'] # it starts with 0
# print(type(roles))
# print(roles)

# List access
# postive indexing
# print(roles[0],roles[1],roles[2]) # admin, user, editor

# negative indexing
# print(roles[-1],roles[-2],roles[-3]) # editor, user, admin

# slicing concept  list[start:end:step]
# print(roles[0:2]) # admin, user

# add element to list
# roles.append('manager') # add element at the end
# print(roles)
# roles.insert(1,'developer') # add element at specific index
# print(roles)

# remove element from list
# roles.pop() # remove last element or specific index
# print(roles)
# roles.remove('developer') # remove element by value
# print(roles)
# roles.clear() #instead of del roles, we can use roles.clear() to remove all elements from list
# print(roles) # []

# Additional methods - print to see output
# print(roles.index('admin')) # 0
# print(roles.count('admin')) # 1
# roles.reverse() # reverse the list
# print(roles)
# roles.sort() # sort the list
# print(roles)
# roles2 = roles.copy() # copy the list

# List loop - in, range(), enumerate()
# for role in roles: 
#     print(role)

# for index, role in enumerate(roles):
#     print(index,role)

# for index in range(len(roles)):
#     print(index,roles[index])

# List joint methods - join(), split()
roles_str = ', '.join(roles)
# print(roles_str) # admin, user, editor

new_roles = roles_str.split(', ')
# print(new_roles) # ['admin', 'user', 'editor']

# List merge - extend()
roles.extend(new_roles)
print(roles) # ['admin', 'user', 'editor', 'admin', 'user', 'editor']