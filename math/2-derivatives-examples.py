# Converted from 2-derivatives-examples.ipynb

# ======================================================================
# # Derivatives in python
# ======================================================================

# ======================================================================
# Derivatives tell us how fast a function is changing at any point. Think of it as `slope` or `steepness` of a curves at a specific point.
# ======================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt

# simple numerical derivatives function
def numerical_derivatives(f, x, h=0.001):
    """Calculate derivative numerically using central difference: (f(x+h) - f(x-h)) / (2h)"""
    return (f(x + h) - f(x - h) / (2 * h))

# ======================================================================
# ## what is deritivates?
# ======================================================================

# ======================================================================
# A derivative measures how much y changes when x changes by a tiny amount. if the function y = 2x, then derivative is always 2 (steepness is constant).
# ======================================================================

# %%
# Example: y = 2x
# The derivative of y = 2x is always 2
# This means: for every step in x, y increases by 2

def f(x):
    return 2 * x

# The derivative at any point is 2.
x_point = 4
derivative_value = 2 # derivative of 2x is 2
print(f"For y = 2x, at x = {x_point}, the derivative (slope) = {derivative_value}")

# visualize function and its derivative
x = np.linspace(0, 10, 100)
y = f(x)
dy_dx = np.full_like(x, 2) # derivative is constant at 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# plot function
ax1.plot(x, y, 'b-', linewidth=2, label='y = 2x')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function: y = 2x')
ax1.grid(True)
ax1.legend()

# plot derivative
ax2.plot(x, dy_dx, 'r-', linewidth=2, label='dy_dx = 2')
ax2.set_xlabel('x')
ax2.set_ylabel('dy/dx')
ax2.set_title("Derivative: dy/dx = 2 (constant)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# ======================================================================
# ## Derivative of y = x²
# ======================================================================

# ======================================================================
# The derivative of y = x² is y' = 2x. This means the slope changes as x changes!
# ======================================================================

# %%
# Example: y = x², derivative = 2x
def f(x):
    return x ** 2

def df_dx(x):
    return 2 * x # derivative of x² is 2x

# Check derivatives at different points
points = [-2, -1, 0, 1, 2, 3]
print("Point | Function value | Derivative (slope)")
print("-" * 45)
for x in points:
    y = f(x)
    slope = df_dx(x)
    print(f"x={x:3d} | y = {y:3d}        | dy/dx = {slope:3d}")

# Visualize
x = np.linspace(-3, 3, 100)
y = f(x)
dy_dx = df_dx(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot function
ax1.plot(x, y, 'b-', linewidth=2, label='y = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function: y = x²')
ax1.grid(True)
ax1.legend()
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot derivative
ax2.plot(x, dy_dx, 'r-', linewidth=2, label="dy/dx = 2x")
ax2.set_xlabel('x')
ax2.set_ylabel('dy/dx')
ax2.set_title('Derivative: dy/dx = 2x')
ax2.grid(True)
ax2.legend()
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# ======================================================================
# ## Visualizing Tangent Lines
# ======================================================================

# ======================================================================
# The derivative gives us the slope of the tangent line by any point. Let's draw the line on the curve
# ======================================================================

# %%
# Draw tangent lines at different points on y = x²
def f(x):
    return x ** 2

def df_dx(x):
    return 2 * x

def tangent_line(x, x0):
    """Equation of tangent line at x0: y = f(x0) + f'(x0)(x - x0)"""
    return f(x0) + df_dx(x0) * (x - x0)

x = np.linspace(-3, 3, 100)
y = f(x)

# Points where we'll draw tangent lines
tangent_points = [-2, -1, 0, 1, 2]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='y = x²')

# Draw tangent lines
for x0 in tangent_points:
    x_tangent = np.linspace(x0 - 1, x0 + 1, 50)
    y_tangent = tangent_line(x_tangent, x0)
    slope = df_dx(x0)
    plt.plot(x_tangent, y_tangent, 'r--', linewidth=1.5, alpha=0.7)
    plt.plot(x0, f(x0), 'ro', markersize=8)
    plt.text(x0 + 0.2, f(x0) + 0.5, f'slope={slope}', fontsize=9)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Tangent Lines on y = x²')
plt.grid(True)
plt.legend()
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# ======================================================================
# ## Common Derivative Rules
# ======================================================================

# ======================================================================
# Here are some simple rules for finding derivatives:
# ======================================================================

# %%
# Rule 1: Derivative of constant is 0
# y = 5 → dy/dx = 0

# Rule 2: Derivative of x^n is n*x^(n-1)
# y = x³ → dy/dx = 3x²
# y = x⁴ → dy/dx = 4x³

# Rule 3: Derivative of constant * function
# y = 3x² → dy/dx = 6x

# Let's verify Rule 2 with examples
print("Function    | Derivative")
print("-" * 35)
print("y = x²      | dy/dx = 2x")
print("y = x³      | dy/dx = 3x²")
print("y = x⁴      | dy/dx = 4x³")
print("y = x⁵      | dy/dx = 5x⁴")
print()
print("Let's check y = x³:")

def f_cubic(x):
    return x ** 3

def df_cubic_dx(x):
    return 3 * x ** 2

test_points = [0, 1, 2, 3]
print("\nPoint | Function | Derivative")
print("-" * 35)
for x in test_points:
    y = f_cubic(x)
    slope = df_cubic_dx(x)
    print(f"x={x}   | y = {y:3d}  | dy/dx = {slope:3d}")

# ======================================================================
# ## Derivative of y = x³
# ======================================================================

# ======================================================================
# Let's visualize the function and its derivative together.
# ======================================================================

# %%
# y = x³, derivative = 3x²
def f(x):
    return x ** 3

def df_dx(x):
    return 3 * x ** 2

x = np.linspace(-2, 2, 100)
y = f(x)
dy_dx = df_dx(x)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot function
ax1.plot(x, y, 'g-', linewidth=2, label='y = x³')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function: y = x³')
ax1.grid(True)
ax1.legend()
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot derivative
ax2.plot(x, dy_dx, 'orange', linewidth=2, label="dy/dx = 3x²")
ax2.set_xlabel('x')
ax2.set_ylabel('dy/dx')
ax2.set_title('Derivative: dy/dx = 3x²')
ax2.grid(True)
ax2.legend()
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# ======================================================================
# ## Derivative of y = 2x + 3
# ======================================================================

# ======================================================================
# The derivative of a sum is the sum of derivatives. y = 2x + 3 → dy/dx = 2 (the constant 3 disappears!)
# ======================================================================

# %%
# y = 2x + 3, derivative = 2
def f(x):
    return 2 * x + 3

def df_dx(x):
    return 2  # derivative of 2x is 2, derivative of 3 is 0

x = np.linspace(-5, 5, 100)
y = f(x)
dy_dx = np.full_like(x, 2)  # constant derivative

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot function
ax1.plot(x, y, 'purple', linewidth=2, label='y = 2x + 3')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function: y = 2x + 3')
ax1.grid(True)
ax1.legend()
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot derivative
ax2.plot(x, dy_dx, 'r-', linewidth=2, label="dy/dx = 2")
ax2.set_xlabel('x')
ax2.set_ylabel('dy/dx')
ax2.set_title('Derivative: dy/dx = 2 (constant)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print("Note: The constant +3 doesn't affect the derivative!")
print("The slope is always 2, regardless of where we are on the line.")

# ======================================================================
# ##  Comparing Functions and Their Derivatives
# ======================================================================

# ======================================================================
# Let's see multiple functions and their derivatives side by side.
# ======================================================================

# %%
x = np.linspace(-3, 3, 100)

# Define functions and their derivatives
functions = [
    (lambda x: 2*x, lambda x: np.full_like(x, 2), 'y = 2x', 'dy/dx = 2', 'blue'),
    (lambda x: x**2, lambda x: 2*x, 'y = x²', 'dy/dx = 2x', 'red'),
    (lambda x: x**3, lambda x: 3*x**2, 'y = x³', 'dy/dx = 3x²', 'green'),
]

fig, axes = plt.subplots(len(functions), 2, figsize=(14, 12))

for i, (f, df, f_label, df_label, color) in enumerate(functions):
    y = f(x)
    dy_dx = df(x)
    
    # Plot function
    axes[i, 0].plot(x, y, color=color, linewidth=2, label=f_label)
    axes[i, 0].set_xlabel('x')
    axes[i, 0].set_ylabel('y')
    axes[i, 0].set_title(f'Function: {f_label}')
    axes[i, 0].grid(True)
    axes[i, 0].legend()
    axes[i, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[i, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Plot derivative
    axes[i, 1].plot(x, dy_dx, color=color, linewidth=2, label=df_label)
    axes[i, 1].set_xlabel('x')
    axes[i, 1].set_ylabel('dy/dx')
    axes[i, 1].set_title(f'Derivative: {df_label}')
    axes[i, 1].grid(True)
    axes[i, 1].legend()
    axes[i, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[i, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# ======================================================================
# ## Key Takeaways
# ======================================================================

# ======================================================================
# **What derivatives tell us:**
# 
# - The slope (steepness) of a function at any point
# - How fast the function is changing
# - Whether the function is increasing or decreasing
# ======================================================================

# ======================================================================
# **Simple rules:**
# 
# - Constant: y = 5 → dy/dx = 0
# - Power: y = xⁿ → dy/dx = nxⁿ⁻¹
# - Constant times function: y = k·f(x) → dy/dx = k·f'(x)
# - Sum: y = f(x) + g(x) → dy/dx = f'(x) + g'(x)
# 
# **Remember:** The derivative is like asking "what's the slope right here?"
# ======================================================================

