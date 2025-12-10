# Converted from pattern.ipynb

# ======================================================================
# 2 basic of pattern
#  1.number of row
#  2. number of col to print space/* etc,.
# ======================================================================

# %%
# square pattern
n = 5
for i in range(n):
    for j in range(n):
        print("*", end=" ")
    print()

# %%
# increasing triangle
for i in range(n):
    for j in range(i + 1):
        print("*", end=" ")
    print()

# %%
# decrease triange
for i in range(n):
    for j in range(i, n):
        print("*", end=" ")
    print()

# %%
# right sided triangle
for i in range(n):
    for j in range(i, n - 1):
        print(" ", end=" ")
    for j in range(i + 1):
        print("*", end=" ")
    print()

# %%
# another right sided triangle
for i in range(n):
    for j in range(i + 1):
        print(" ", end=" ")
    for j in range(i, n):
        print("*", end=" ")
    print()

# %%
# hill pattern
for i in range(n):
    for j in range(i, n):
        print(" ", end=" ")
    for j in range(
        i
    ):  # remember one star loop in loops in outer like i times and last loops doesn't loop move next loop
        print("*", end=" ")
    for j in range(i + 1):
        print("*", end=" ")
    print()

# %%
# reserve hill pattern
for i in range(n):
    for j in range(i + 1):
        print(" ", end=" ")
    for j in range(i, n - 1):
        print("*", end=" ")
    for j in range(i, n):
        print("*", end=" ")
    print()

# %%
# diamond pattern
for i in range(n - 1):
    for j in range(i, n):
        print(" ", end=" ")
    for j in range(i):
        print("*", end=" ")
    for j in range(i + 1):
        print("*", end=" ")
    print()
for i in range(n):
    for j in range(i + 1):
        print(" ", end=" ")
    for j in range(i, n - 1):
        print("*", end=" ")
    for j in range(i, n):
        print("*", end=" ")
    print()

