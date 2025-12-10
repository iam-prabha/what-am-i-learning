# Converted from 03-functions.ipynb

# ======================================================================
# # Understanding Python Functions - Beginner Friendly Guide
# Welcome! In this notebook, we will learn about **Python functions** in a simple and easy way.
# ======================================================================

# ======================================================================
# ## 1️⃣ What is a Function?
# A **function** is a block of reusable code that performs a specific task. Instead of writing the same code multiple times, we can **define a function** and **call it whenever needed**.
# 
# ### Example of a function in real life:
# - Think of a **blender** 🍹. It takes **ingredients** (inputs), blends them, and gives you a **smoothie** (output).
# ======================================================================

# ======================================================================
# ## 2️⃣ Why Use Functions?
# - **Avoid repetition**: Instead of writing the same code again and again, we write a function once and reuse it.
# - **Better organization**: Helps in structuring your code neatly.
# - **Easier debugging**: Fix issues in one place instead of multiple locations.
# ======================================================================

# ======================================================================
# ## 3️⃣ How to Create and Call a Function?
# ======================================================================

# %%
# Defining a function
def say_hello():
    print("Hello, World!")


# Calling the function
say_hello()

# ======================================================================
# ## 4️⃣ Function Arguments and Return Values
# ======================================================================

# %%
# Function with parameters and return value
def add_numbers(a, b):
    return a + b


# Calling the function
result = add_numbers(5, 3)
print("Sum:", result)

# ======================================================================
# ## 5️⃣ Default and Keyword Arguments
# ======================================================================

# %%
# Function with default argument
def greet(name="Guest"):
    print("Hello,", name)


greet("Alice")  # Using custom argument
greet()  # Using default argument

# ======================================================================
# ## 6️⃣ Lambda Functions
# ======================================================================

# %%
# Lambda function (Anonymous function)
multiply = lambda x, y: x * y
print("Multiplication:", multiply(4, 5))

# ======================================================================
# ## 7️⃣ Scope of Variables in Functions
# ======================================================================

# %%
# Example of variable scope
def example():
    local_var = "I'm inside the function"
    print(local_var)


example()
# print(local_var)  # This will give an error because local_var is not accessible outside the function

# ======================================================================
# ## 8️⃣ Practical Example
# ======================================================================

# %%
# Function to check if a number is even or odd
def check_even_odd(number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"


# Test the function
num = 7
print(f"{num} is", check_even_odd(num))

