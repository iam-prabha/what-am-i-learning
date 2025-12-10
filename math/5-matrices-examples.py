# Converted from 5-matrices-examples.ipynb

# ======================================================================
# # Matrices in Python
# ======================================================================

# ======================================================================
# Matrices are likes tables of numbers: rows and columns arranged in a grid. Think of them as way to organize and transform data - essential for neural networks! 
# ======================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# ## What is a Matrix?
# ======================================================================

# ======================================================================
# A matrix is a rectangular array of numbers arranged in rows and columns. We write it as: A = [[a₁₁, a₁₂], [a₂₁, a₂₂]]§ The first number is the row, the second is the column!
# ======================================================================

# %%
# Examples matrices
# A 2x2 matrix (2 rows, 2 columns)
A = np.array([[1, 2],
              [3, 4]])

# A 2x3 matrix (2 rows, 3 columns)
B = np.array([[1, 2, 3],
              [4, 5, 6]])

# A 3x2 matrix (3 rows, 2 columns)
C = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print("Matrix A (2x2):")
print(A)
print(f"\nShape: {A.shape}  (rows={A.shape[0]}, columns={A.shape[1]})")
print()

print("Matrix B (2x3):")
print(B)
print(f"\nShape: {B.shape}  (rows={B.shape[0]}, columns={B.shape[1]})")
print()

print("Matrix C (3x2):")
print(C)
print(f"\nShape: {C.shape}  (rows={C.shape[0]}, columns={C.shape[1]})")

# ======================================================================
# ## Accessing Matrix Elements
# ======================================================================

# ======================================================================
# You can access individual elements using row and column indices. Remember: Python starts counting from 0!
# ======================================================================

# %%
# Example: Accessing matrix elements
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print("Matrix A:")
print(A)
print(f"\nShape: {A.shape}")
print()

# Accessing elements
print("Accessing elements:")
print(f"A[0, 0] = {A[0, 0]}  (first row, first column)")
print(f"A[0, 1] = {A[0, 1]}  (first row, second column)")
print(f"A[1, 0] = {A[1, 0]}  (second row, first column)")
print(f"A[1, 1] = {A[1, 1]}  (second row, second column)")
print(f"A[2, 0] = {A[2, 0]}  (third row, first column)")
print()

# Accessing entire rows
print("Accessing rows:")
print(f"A[0, :] = {A[0, :]}  (first row)")
print(f"A[1, :] = {A[1, :]}  (second row)")
print()

# Accessing entire columns
print("Accessing columns:")
print(f"A[:, 0] = {A[:, 0]}  (first column)")
print(f"A[:, 1] = {A[:, 1]}  (second column)")

# ======================================================================
# ## Matrix Addition
# ======================================================================

# ======================================================================
# Adding two matrices: add corresponding elements! Both matrices must have the same shape (same number of rows and columns).
# ======================================================================

# %%
# Example: A + B
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matrix addition: add corresponding elements
result = A + B

print("Matrix A:")
print(A)
print()

print("Matrix B:")
print(B)
print()

print("A + B (add corresponding elements):")
print(result)
print()

# Show step by step
print("Step by step:")
print(f"A[0, 0] + B[0, 0] = {A[0, 0]} + {B[0, 0]} = {result[0, 0]}")
print(f"A[0, 1] + B[0, 1] = {A[0, 1]} + {B[0, 1]} = {result[0, 1]}")
print(f"A[1, 0] + B[1, 0] = {A[1, 0]} + {B[1, 0]} = {result[1, 0]}")
print(f"A[1, 1] + B[1, 1] = {A[1, 1]} + {B[1, 1]} = {result[1, 1]}")

# ======================================================================
# ## Scalar Multiplication
# ======================================================================

# ======================================================================
# Multiplying a matrix by a number: multiply every element by that number! This scales the entire matrix
# ======================================================================

# %%
# Example: 2 * A, 0.5 * A, -1 * A
A = np.array([[1, 2],
              [3, 4]])

A_doubled = 2 * A      # Multiply every element by 2
A_half = 0.5 * A       # Multiply every element by 0.5
A_negated = -1 * A     # Multiply every element by -1

print("Original matrix A:")
print(A)
print()

print("2 * A (double every element):")
print(A_doubled)
print()

print("0.5 * A (half every element):")
print(A_half)
print()

print("-1 * A (negate every element):")
print(A_negated)
print()

# Show step by step for 2 * A
print("Step by step for 2 * A:")
print(f"2 * {A[0, 0]} = {A_doubled[0, 0]}")
print(f"2 * {A[0, 1]} = {A_doubled[0, 1]}")
print(f"2 * {A[1, 0]} = {A_doubled[1, 0]}")
print(f"2 * {A[1, 1]} = {A_doubled[1, 1]}")

# ======================================================================
# ## Matrix-Vector Multiplication
# ======================================================================

# ======================================================================
# Multiplying a matrix by a vector: transform the vector! This is the most important operation for neural networks. For A (m×n) × v (n×1), we get a result (m×1).
# ======================================================================

# %%
# Example: A × v
# A is 2×3, v is 3×1, result is 2×1
A = np.array([[1, 2, 3],
              [4, 5, 6]])

v = np.array([2, 3, 4])

# Matrix-vector multiplication
result = A @ v  # or np.dot(A, v)

print("Matrix A (2×3):")
print(A)
print()

print("Vector v (3×1):")
print(v)
print()

print("A × v (2×1 result):")
print(result)
print()

# Show step by step
print("Step by step calculation:")
print("For each row in A, multiply by v and sum:")
print()

row1 = A[0, :]
row2 = A[1, :]

print(f"First row of result:")
print(f"  A[0, :] · v = {row1[0]}*{v[0]} + {row1[1]}*{v[1]} + {row1[2]}*{v[2]}")
print(f"              = {row1[0]*v[0]} + {row1[1]*v[1]} + {row1[2]*v[2]}")
print(f"              = {result[0]}")
print()

print(f"Second row of result:")
print(f"  A[1, :] · v = {row2[0]}*{v[0]} + {row2[1]}*{v[1]} + {row2[2]}*{v[2]}")
print(f"              = {row2[0]*v[0]} + {row2[1]*v[1]} + {row2[2]*v[2]}")
print(f"              = {result[1]}")

# ======================================================================
# ## Matrix-Matrix Multiplication
# ======================================================================

# ======================================================================
# Multiplying two matrices: A (m×n) × B (n×p) = C (m×p) The number of columns in A must equal the number of rows in B! Each element of C is the dot product of a row from A and a column from B.
# ======================================================================

# %%
# Example: A × B
# A is 2×3, B is 3×2, result is 2×2
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Matrix-matrix multiplication
result = A @ B  # or np.dot(A, B)

print("Matrix A (2×3):")
print(A)
print()

print("Matrix B (3×2):")
print(B)
print()

print("A × B (2×2 result):")
print(result)
print()

# Show step by step
print("Step by step calculation:")
print("Each element C[i, j] = (row i of A) · (column j of B)")
print()

print(f"C[0, 0] = A[0, :] · B[:, 0]")
print(f"        = [{A[0, 0]}, {A[0, 1]}, {A[0, 2]}] · [{B[0, 0]}, {B[1, 0]}, {B[2, 0]}]")
print(f"        = {A[0, 0]}*{B[0, 0]} + {A[0, 1]}*{B[1, 0]} + {A[0, 2]}*{B[2, 0]}")
print(f"        = {A[0, 0]*B[0, 0]} + {A[0, 1]*B[1, 0]} + {A[0, 2]*B[2, 0]}")
print(f"        = {result[0, 0]}")
print()

print(f"C[0, 1] = A[0, :] · B[:, 1]")
print(f"        = [{A[0, 0]}, {A[0, 1]}, {A[0, 2]}] · [{B[0, 1]}, {B[1, 1]}, {B[2, 1]}]")
print(f"        = {A[0, 0]}*{B[0, 1]} + {A[0, 1]}*{B[1, 1]} + {A[0, 2]}*{B[2, 1]}")
print(f"        = {A[0, 0]*B[0, 1]} + {A[0, 1]*B[1, 1]} + {A[0, 2]*B[2, 1]}")
print(f"        = {result[0, 1]}")
print()

print(f"C[1, 0] = A[1, :] · B[:, 0]")
print(f"        = [{A[1, 0]}, {A[1, 1]}, {A[1, 2]}] · [{B[0, 0]}, {B[1, 0]}, {B[2, 0]}]")
print(f"        = {A[1, 0]}*{B[0, 0]} + {A[1, 1]}*{B[1, 0]} + {A[1, 2]}*{B[2, 0]}")
print(f"        = {A[1, 0]*B[0, 0]} + {A[1, 1]*B[1, 0]} + {A[1, 2]*B[2, 0]}")
print(f"        = {result[1, 0]}")
print()

print(f"C[1, 1] = A[1, :] · B[:, 1]")
print(f"        = [{A[1, 0]}, {A[1, 1]}, {A[1, 2]}] · [{B[0, 1]}, {B[1, 1]}, {B[2, 1]}]")
print(f"        = {A[1, 0]}*{B[0, 1]} + {A[1, 1]}*{B[1, 1]} + {A[1, 2]}*{B[2, 1]}")
print(f"        = {A[1, 0]*B[0, 1]} + {A[1, 1]*B[1, 1]} + {A[1, 2]*B[2, 1]}")

# ======================================================================
# ## Matrix Transpose
# ======================================================================

# ======================================================================
# Transposing a matrix: flip rows and columns! A transpose turns a m×n matrix into an n×m matrix. Notation: Aᵀ
# ======================================================================

# %%
# Example: A transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_transpose = A.T  # or np.transpose(A)

print("Original matrix A (2×3):")
print(A)
print()

print("A transpose (3×2):")
print(A_transpose)
print()

# Show what happens
print("What happened:")
print(f"Row 0 of A [{A[0, 0]}, {A[0, 1]}, {A[0, 2]}] became column 0 of Aᵀ")
print(f"Row 1 of A [{A[1, 0]}, {A[1, 1]}, {A[1, 2]}] became column 1 of Aᵀ")
print()

print("Step by step:")
print(f"A[0, 0] = {A[0, 0]} → Aᵀ[0, 0] = {A_transpose[0, 0]}")
print(f"A[0, 1] = {A[0, 1]} → Aᵀ[1, 0] = {A_transpose[1, 0]}")
print(f"A[0, 2] = {A[0, 2]} → Aᵀ[2, 0] = {A_transpose[2, 0]}")
print(f"A[1, 0] = {A[1, 0]} → Aᵀ[0, 1] = {A_transpose[0, 1]}")
print(f"A[1, 1] = {A[1, 1]} → Aᵀ[1, 1] = {A_transpose[1, 1]}")
print(f"A[1, 2] = {A[1, 2]} → Aᵀ[2, 1] = {A_transpose[2, 1]}")
print()

print("Notice: (Aᵀ)ᵀ = A (transposing twice gives you back the original!)")
A_transpose_transpose = A_transpose.T
print(f"(Aᵀ)ᵀ = \n{A_transpose_transpose}")

# ======================================================================
# ## Identity Matrix
# ======================================================================

# ======================================================================
# The identity matrix I is special: when you multiply any matrix by I, you get back the same matrix! I has 1s on the diagonal and 0s everywhere else. A × I = A and I × A = A
# ======================================================================

# %%
# Example: Identity matrix
# 2×2 identity matrix
I2 = np.eye(2)  # or np.identity(2)

# 3×3 identity matrix
I3 = np.eye(3)

print("2×2 Identity matrix I:")
print(I2)
print()

print("3×3 Identity matrix I:")
print(I3)
print()

# Show that A × I = A
A = np.array([[1, 2],
              [3, 4]])

result = A @ I2

print("Matrix A:")
print(A)
print()

print("I × A (should equal A):")
print(result)
print()

print(f"A == I × A? {np.array_equal(A, result)}")
print()

# Show with a different matrix
B = np.array([[5, 6, 7],
              [8, 9, 10]])

print("Matrix B (2×3):")
print(B)
print()

print("B × I₃ (should equal B):")
print(B @ I3)
print()

print(f"B == B × I₃? {np.array_equal(B, B @ I3)}")
print()

print("Notice: The identity matrix acts like the number 1 in multiplication!")

# ======================================================================
# ## Matrix Properties
# ======================================================================

# ======================================================================
# Some important properties of matrices:
# 
# - A + B = B + A (addition is commutative)
# - (A × B) × C = A × (B × C) (multiplication is associative)
# - A × (B + C) = A × B + A × C (distributive)
# - But A × B ≠ B × A in general! (multiplication is NOT commutative)
# ======================================================================

# %%
# Example: Matrix properties
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = np.array([[1, 0],
              [0, 1]])

print("Matrices:")
print("A =")
print(A)
print()

print("B =")
print(B)
print()

print("C =")
print(C)
print()

# Property 1: A + B = B + A
print("Property 1: A + B = B + A")
print(f"A + B = \n{A + B}")
print(f"B + A = \n{B + A}")
print(f"Equal? {np.array_equal(A + B, B + A)}")
print()

# Property 2: (A × B) × C = A × (B × C)
print("Property 2: (A × B) × C = A × (B × C)")
left = (A @ B) @ C
right = A @ (B @ C)
print(f"(A × B) × C = \n{left}")
print(f"A × (B × C) = \n{right}")
print(f"Equal? {np.array_equal(left, right)}")
print()

# Property 3: A × (B + C) = A × B + A × C
print("Property 3: A × (B + C) = A × B + A × C")
left = A @ (B + C)
right = A @ B + A @ C
print(f"A × (B + C) = \n{left}")
print(f"A × B + A × C = \n{right}")
print(f"Equal? {np.array_equal(left, right)}")
print()

# Property 4: A × B ≠ B × A (in general)
print("Property 4: A × B ≠ B × A (multiplication is NOT commutative!)")
AB = A @ B
BA = B @ A
print(f"A × B = \n{AB}")
print(f"B × A = \n{BA}")
print(f"Equal? {np.array_equal(AB, BA)}")
print("Notice: They're different! Order matters in matrix multiplication!")

# ======================================================================
# ## Common Matrix Operations Summary
# ======================================================================

# ======================================================================
# Let's see different matrix operations side by side!
# ======================================================================

# %%
# Summary: Different matrix operations
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

v = np.array([2, 3])

print("Matrix A:")
print(A)
print()

print("Matrix B:")
print(B)
print()

print("Vector v:")
print(v)
print()

print("=" * 50)
print("OPERATIONS:")
print("=" * 50)
print()

print("1. Addition: A + B")
print(A + B)
print()

print("2. Scalar multiplication: 3 * A")
print(3 * A)
print()

print("3. Matrix-vector multiplication: A @ v")
print(A @ v)
print()

print("4. Matrix-matrix multiplication: A @ B")
print(A @ B)
print()

print("5. Transpose: A.T")
print(A.T)
print()

print("6. Element-wise multiplication (NOT matrix multiplication!): A * B")
print(A * B)
print("(This multiplies each corresponding element separately)")

