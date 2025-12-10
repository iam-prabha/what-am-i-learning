# Converted from 10-exception_handling.ipynb

# ======================================================================
# # Python File Handling & Exception Handling
# 
# This notebook covers **File Handling** and **Exception Handling** in Python.
# 
# ======================================================================

# ======================================================================
# ## File Handling in Python
# 
# Python allows you to read, write, and manipulate files using the `open()` function.
# ======================================================================

# %%
# Opening and Reading a File
with open("sample.txt", "r") as file:
    content = file.read()
    print(content)  # Display file content

# ======================================================================
# ### Writing to a File
# ======================================================================

# %%
# Writing to a file
with open("sample.txt", "w") as file:
    file.write("Hello, this is a new file!")

# ======================================================================
# ### Appending to a File
# ======================================================================

# %%
# Appending to a file
with open("sample.txt", "a") as file:
    file.write("\nAppending new content!")

# ======================================================================
# ## Exception Handling in Python
# 
# Python provides mechanisms to handle runtime errors using `try-except` blocks.
# ======================================================================

# %%
# Handling division by zero error
try:
    result = 10 / 0  # This will cause ZeroDivisionError
except ZeroDivisionError:
    print("Error: Division by zero is not allowed!")

# ======================================================================
# ### Catching Multiple Exceptions
# ======================================================================

# %%
try:
    num = int(input("Enter a number: "))
    result = 10 / num
    print(result)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input! Please enter a number.")

# ======================================================================
# ### Using `finally` Block
# ======================================================================

# %%
try:
    file = open("sample.txt", "r")
    content = file.read()
    print(content)
except FileNotFoundError:
    print("File not found!")
finally:
    print("Execution completed.")

# ======================================================================
# ### Raising Custom Exceptions
# ======================================================================

# %%
def check_age(age):
    if age < 18:
        raise ValueError("Age must be 18 or older!")
    print("Access granted")


try:
    check_age(16)
except ValueError as e:
    print("Error:", e)

