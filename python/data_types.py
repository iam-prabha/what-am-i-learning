#data types
item_id = 1 # int
item_name = "rank" # "" or ''
item_quantity = 10.67 # float
item_restocked = True # boolean

# print data types to see the type of data
type(item_id) # int
type(item_name) # str
type(item_quantity) # float
type(item_restocked) # bool

# variable declaration
# variable_name is a case sensitive like item_name and Item_name are different

# Dynamic Typing
print(item_id) # 1
type(item_id) # int

# type conversion
# string can't be converted to int or float
# int to string
item_id = str(item_id) # "1"
# string to float
item_quantity = int(item_quantity) # 10
# boolean to string
item_restocked = str(item_restocked) # "True"
# string to boolean
item_restocked = bool(item_restocked) # True

# arithmetic operations
# addition -> +
# subtraction -> -
# multiplication -> *
# division -> /
# modulus -> %
# exponentiation(power) -> ** like 2**3 = 8 | (2^3) = 8 are same
# floor division -> // like 5//2 = 2 | 5/2 = 2.5 are same

# bitwise operations
# AND -> &
# OR -> |
# XOR -> ^
# NOT -> ~
# Left shift -> <<
# Right shift -> >>

# comparison operations
# equal -> == like 5 == 5 = True
# not equal -> != like 5 != 5 = False
# greater than -> > like 5 > 5 = False
# less than -> < like 5 < 5 = False
# greater than or equal to -> >= like 5 >= 5 = True
# less than or equal to -> <= like 5 <= 5 = True