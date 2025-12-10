# Converted from 1-math-functions-examples.ipynb

# ======================================================================
# # Simple Math Functions in python
# ======================================================================

# ======================================================================
# A function is like machine: you put number of a (x) and get out another number (y).
# ======================================================================

# %%
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# ## Linear Functions: y = X * K
#
# The simplest function Multiply x by a constant number k.
# ======================================================================

# %%
# Example: y = 2 * x
# If x = 1, then y = 2
# If x = 2, then y = 4
# If x = 3, then y = 6

x = 1
y = 2 * x
print(f"when x = {x}, y = {y}")

x = 5
y = 2
print(f"when x = {x}, y = {y}")

# Let's plot it
x_values = np.linspace(0, 10, 100)
y_values = 2 * x_values

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "b-", linewidth=2, label="y = 2x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Function: y = 2x")
plt.grid(True)
plt.legend()
plt.show()

# %%
# Different values of k
k_values = [1, 2, 3, 0.5, -1]
x = np.linspace(-5, 5, 100)
print(x)

plt.figure(figsize=(8, 5))
for k in k_values:
    y = k * x
    plt.plot(x, y, label=f"y={k}x", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Different Linear Functions: y = kx")
plt.grid(True)
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)

# ======================================================================
# ## Quadratic Functions: y = x²
# ======================================================================

# ======================================================================
# Sqaure the input number. if x = 2, then y = 4. if x = 3, then y = 9.
# ======================================================================

# %%
# Example: y = x²
examples = [-3, -2, -1, 0, 1, 2, 3]
for x in examples:
    y = x**2
    print(f"when x = {x}, y = x² = {y}")

# plot it
x_values = np.linspace(-5, 5, 100)
y_values = x_values**2

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "r-", linewidth=2, label="y = x²")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Functions: y = x²")
plt.grid()
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.show()

# ======================================================================
# ## Cubic Functions: y = x³
# ======================================================================

# ======================================================================
# Cube the input number. if  x = 2, then y = 8. if x = 3, then y = 27.
# ======================================================================

# %%
# Example: y = x³
examples = [-2, -1, 0, 1, 2]
for x in examples:
    y = x**3
    print(f"When x = {x}, y = x³ = {y}")

# Plot it
x_values = np.linspace(-3, 3, 100)
y_values = x_values**3

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "g-", linewidth=2, label="y = x³")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic Function: y = x³")
plt.grid(True)
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.show()

# ======================================================================
# ## square Root Function: y = √x
# ======================================================================

# ======================================================================
# The opposite of squaring. if x = 4, then y = 2. if x = 9, then y = 3.
# ======================================================================

# %%
# Example: y = √x
examples = [0, 1, 4, 9, 16, 25]
for x in examples:
    y = np.sqrt(x)
    print(f"When x = {x}, y = √{x} = {y}")

# Plot it
x_values = np.linspace(0, 25, 100)
y_values = np.sqrt(x_values)

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "m-", linewidth=2, label="y = √x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Square Root Function: y = √x")
plt.grid(True)
plt.legend()
plt.show()

# ======================================================================
# ## Exponential Function: y = 2^x
# ======================================================================

# ======================================================================
# Double the result each time increases by 1. if x = 1, then y = 2. if x = 3, then y = 8.
# ======================================================================

# %%
# Example: y = 2^x
examples = [0, 1, 2, 3, 4, 5]
for x in examples:
    y = 2**x
    print(f"When x = {x}, y = 2^{x} = {y}")

# Plot it
x_values = np.linspace(-2, 5, 100)
y_values = 2**x_values

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "c-", linewidth=2, label="y = 2^x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exponential Function: y = 2^x")
plt.grid(True)
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.show()

# ======================================================================
# ## Sine Function: y = sin(x)
# ======================================================================

# ======================================================================
# A wavy function that repeats. used for waves, circle, and oscillations.
# ======================================================================

# %%
# Example: y = sin(x)
import math

examples = [0, math.pi / 2, math.pi, 3 * math.pi, 2 * math.pi]
for x in examples:
    y = math.sin(x)
    print(f"when x = {x:.2f}, y = sin(x) = {y:.2f}")

# Plot it
x_values = np.linspace(0, 4 * np.pi, 100)
y_values = np.sin(x_values)

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, "orange", linewidth=2, label="y = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine Function: y = sin(x)")
plt.grid(True)
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.show()

# ======================================================================
# ## Combined Functions: y = 2x + 3
# ======================================================================

# ======================================================================
# You can combine operations! Multiply x by 2, then add 3.
# ======================================================================

# %%
# Example: y = 2x + 3
examples = [0, 1, 2, 3, 4]
for x in examples:
    y = 2 * x + 3
    print(f"When x = {x}, y = 2({x}) + 3 = {y}")

# Plot it
x_values = np.linspace(-5, 5, 100)
y_values = 2 * x_values + 3

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, "purple", linewidth=2, label="y = 2x + 3")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Combined Function: y = 2x + 3")
plt.grid(True)
plt.legend()
plt.axhline(y=0, color="k", linewidth=0.5)
plt.axvline(x=0, color="k", linewidth=0.5)
plt.show()

# ======================================================================
# ## summary: compare All Functions
# ======================================================================

# %%
x = np.linspace(0, 5, 100)

plt.figure(figsize=(12, 8))

# Linear
plt.subplot(2, 2, 1)
plt.plot(x, 2 * x, "b-", linewidth=2)
plt.title("y = 2x (Linear)")
plt.grid(True)

# Quadratic
plt.subplot(2, 2, 2)
plt.plot(x, x**2, "r-", linewidth=2)
plt.title("y = x² (Quadratic)")
plt.grid(True)

# Exponential
plt.subplot(2, 2, 3)
plt.plot(x, 2**x, "g-", linewidth=2)
plt.title("y = 2^x (Exponential)")
plt.grid(True)

# Square root
plt.subplot(2, 2, 4)
plt.plot(x, np.sqrt(x), "m-", linewidth=2)
plt.title("y = √x (Square Root)")
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
