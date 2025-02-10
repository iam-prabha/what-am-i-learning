# loops

# inbuilt function range() is used to generate a sequence of numbers
# range(5) # generates a sequence of numbers from 0 to 4
# for loop
# for i in range(5):
#     print(i)

# while loop
# sequence is not determined use while loop
# day = 6
# while day < 7:
#     print("It's day", day)
#     break

# conditional statements
# if statement
# if day == 6:
#     print("It's Saturday")
# else:
#     print("It's not Saturday")

# if-elif-else statement
# if day == 6:
#     print("It's Saturday") 
# elif day == 7:
#     print("It's Sunday")
# else:
#     print("It's not Saturday or Sunday")

# nested if statement
# if day == 6:
#     if day == 7:
#         print("It's Sunday")
#     else:
#         print("It's Saturday")
# else:
#     print("It's not Saturday or Sunday")

# in operator 
# checks if a value is present in a sequence
# days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] # list of days
# if 'Saturday' in days:
#     print("It's Saturday")
# else:
#     print("It's not Saturday")

# not in operator
# checks if a value is not present in a sequence
# if 'Saturday' not in days:
#     print("It's not Saturday")
# else:
#     print("It's Saturday")

# is operator
# checks if two variables are pointing to the same object
# a = 5
# b = 5
# if a is b:
#     print("a and b are pointing to the same object")
# else:
#     print("a and b are not pointing to the same object")

# is not operator
# checks if two variables are not pointing to the same object
# a = 5
# b = 6
# if a is not b:
#     print("a and b are not pointing to the same object")
# else:
#     print("a and b are pointing to the same object")

# all() function
# returns True if all elements in a sequence are True
# otherwise returns False
# a = [True, True, True]
# if all(a):
#     print("All elements are True")
# else:
#     print("All elements are not True")

# break statement
# used to exit a loop
# for i in range(5):
#     if i == 3:
#         break
#     print(i)

# continue statement
# used to skip the current iteration of a loop
# for i in range(5):
#     if i == 3:
#         continue
#     print(i)

# pass statement
# used to do nothing avoid getting an error
a = 5
b = 2 # change the value of b to 6 to see the output
if a < b:
    pass
else:
    print("a is not less than b")