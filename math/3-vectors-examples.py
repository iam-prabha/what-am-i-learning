# Converted from 3-vectors-examples.ipynb

# ======================================================================
# # Vectors in python
# ======================================================================

# ======================================================================
# Vector are like arrows:they have both a direction and a length (magnitude). Think of them as moving a certain distance in a certain direction!
# ======================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# ## what is a vector?
# ======================================================================

# ======================================================================
# A vector is a represented as list of numbers: [x, y] for 2D, [x, y, z] for 3D.
# Each number tells you how far to move in that direction.
# ======================================================================

# %%
# Example vectors in 2D
v1 = np.array([3, 4])  # move 3 units right, 4 units up
v2 = np.array([-2, 1]) # move 2 units left, 1 units up
v3 = np.array([0, 5])  # move o units right, 5 units up (straight up)

print("Vector 1:", v1)
print("Vector 2:", v2)
print("Vector 3:", v3)

# visualize them
fig, ax = plt.subplots(figsize=(10, 8))

# Draw vector from origin
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
            color='b', width=0.005, label=f'v1 = {v1}')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
            color='r', width=0.005, label=f'v2 = {v2}')
ax.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1,
            color='g', width=0.005, label=f'v3 = {v3}')

ax.set_xlim(-5, 5)
ax.set_ylim(-1, 6)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Vector as Arrows')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# ======================================================================
# ## Vector Addition
# ======================================================================

# ======================================================================
# Adding vectors means following one vector, then the other. Result: move the total distance in the combined direction!
# ======================================================================

# %%
# Example : v1 + v2
v1 = np.array([3, 2])
v2 = np.array([1, 3])

# vector addition: just add the components!
result = v1 + v2
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {result}")

# visualize
fig, ax = plt.subplots(figsize=(10, 8))

# Draw v1
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
            color='b', width=0.005, label=f"v1 = {v1}")

# Draw v2 starting from end of v1
ax.quiver(v1[0], v1[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
            color='r', width=0.005, label=f"v2 = {v2}")

# Draw result (v1+v2) from origin
ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1,
            color='g', width=0.008, label=f"v1 + v2 = {result}")

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Vector Addition: v1 + v2')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print("\nNotice: v1 + v2 goes from origin to the end of v2 ( when v2 starts from v1 )")

# %%
# Example: 2 * v, 0.5 * v, -1 * v
v = np.array([2, 1])

v_doubled = 2 * v      # Double the length
v_half = 0.5 * v       # Half the length
v_flipped = -1 * v     # Same length, opposite direction

print(f"Original vector: v = {v}")
print(f"2 * v = {v_doubled}  (double length)")
print(f"0.5 * v = {v_half}  (half length)")
print(f"-1 * v = {v_flipped}  (flipped direction)")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))

vectors = [
    (v, 'b', f'v = {v}'),
    (v_doubled, 'r', f'2v = {v_doubled}'),
    (v_half, 'g', f'0.5v = {v_half}'),
    (v_flipped, 'orange', f'-v = {v_flipped}'),
]

for vec, color, label in vectors:
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
              color=color, width=0.006, label=label)

ax.set_xlim(-3, 5)
ax.set_ylim(-2, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scalar Multiplication')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# ======================================================================
# ## Vector Magnitude (Length)
# ======================================================================

# ======================================================================
# The magnitude (length) of the vector v = [x ,y] is: √(x² + y²) This comes from thr pythagorean theorem!
# ======================================================================

# %%
# Example: magnitude of vectors
v1 = np.array([3, 4])
v2 = np.array([5, 0])
v3 = np.array([1, 1])

# Calculate magnitude
mag1 = np.linalg.norm(v1)  # or np.sqrt(v1[0]**2 + v1[1]**2)
mag2 = np.linalg.norm(v2)
mag3 = np.linalg.norm(v3)

print(f"v1 = {v1}")
print(f"|v1| = √({v1[0]}² + {v1[1]}²) = √{v1[0]**2 + v1[1]**2} = {mag1:.2f}")
print()

print(f"v2 = {v2}")
print(f"|v2| = √({v2[0]}² + {v2[1]}²) = √{v2[0]**2 + v2[1]**2} = {mag2:.2f}")
print()

print(f"v3 = {v3}")
print(f"|v3| = √({v3[0]}² + {v3[1]}²) = √{v3[0]**2 + v3[1]**2} = {mag3:.2f}")

# Visualize with magnitude labels
fig, ax = plt.subplots(figsize=(10, 8))

vectors = [(v1, 'b', mag1), (v2, 'r', mag2), (v3, 'g', mag3)]

for vec, color, mag in vectors:
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
              color=color, width=0.006, label=f'{vec}, |v| = {mag:.2f}')

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Vector Magnitudes')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# ======================================================================
# ## Dot Product
# ======================================================================

# ======================================================================
# The dot product of two vectors: v1 * v2 = x1x2 + y1y2. It tells us how "aligned" the vectors are!
# 
# - Positive = pointing in similar direction
# - Zero = perpendicular (90 degrees)
# - Negative = pointing in opposite directions
# ======================================================================

# %%
# Example: dot product
v1 = np.array([3, 4])
v2 = np.array([1, 2])
v3 = np.array([-4, 3])  # Perpendicular to v1
v4 = np.array([-3, -4])  # Opposite to v1

# Calculate dot products
dot1 = np.dot(v1, v2)      # or v1 @ v2
dot2 = np.dot(v1, v3)
dot3 = np.dot(v1, v4)

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 · v2 = {v1[0]}*{v2[0]} + {v1[1]}*{v2[1]} = {v1[0]*v2[0]} + {v1[1]*v2[1]} = {dot1}")
print()

print(f"v1 = {v1}")
print(f"v3 = {v3}")
print(f"v1 · v3 = {v1[0]}*{v3[0]} + {v1[1]}*{v3[1]} = {v1[0]*v3[0]} + {v1[1]*v3[1]} = {dot2} (perpendicular!)")
print()

print(f"v1 = {v1}")
print(f"v4 = {v4}")
print(f"v1 · v4 = {v1[0]}*{v4[0]} + {v1[1]}*{v4[1]} = {v1[0]*v4[0]} + {v1[1]*v4[1]} = {dot3} (opposite!)")

# Visualize
fig, ax = plt.subplots(figsize=(12, 10))

vectors = [
    (v1, 'b', 'v1'),
    (v2, 'r', f'v2 (dot = {dot1:.1f})'),
    (v3, 'g', f'v3 (dot = {dot2:.1f}, perpendicular)'),
    (v4, 'orange', f'v4 (dot = {dot3:.1f}, opposite)'),
]

for vec, color, label in vectors:
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
              color=color, width=0.006, label=label)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Dot Product Examples')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# ======================================================================
# ## Vector Subtraction
# ======================================================================

# ======================================================================
# Substracting vectors: v1 - v2 = v1 + (-v2) The result points from v2 to v1!
# ======================================================================

# %%
# Example: v1 - v2
v1 = np.array([4, 3])
v2 = np.array([1, 2])

# Vector subtraction
result = v1 - v2
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 - v2 = {result}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))

# Draw v1 and v2 from origin
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
          color='b', width=0.006, label=f'v1 = {v1}')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
          color='r', width=0.006, label=f'v2 = {v2}')

# Draw result from origin
ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1, 
          color='g', width=0.008, label=f'v1 - v2 = {result}')

# Also draw result starting from v2 (shows it goes from v2 to v1) using a line
ax.plot([v2[0], v1[0]], [v2[1], v1[1]], 'g--', linewidth=1.5, alpha=0.6, 
        label='v1 - v2 (from v2 to v1)')
ax.quiver(v2[0], v2[1], result[0], result[1], angles='xy', scale_units='xy', scale=1, 
          color='g', width=0.006, alpha=0.6)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Vector Subtraction: v1 - v2')
ax.grid(True)
ax.legend()
ax.set_aspect('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

