# Converted from 06-numpy.ipynb

# ======================================================================
# # Libraries
# ======================================================================

# %%
import numpy as np

# ======================================================================
# # Create Array
# ======================================================================

# %%
egg_box = np.array(["chicken_egg", "country_egg", "quail_egg"])

# %%
egg_box

# %%
egg_box.shape

# %%
egg_box.ndim

# ======================================================================
# # Array dimensions (0-d, 1-d, 2-d, 3d)
# ======================================================================

# %%
egg_box_zero = np.array("egg")

# %%
egg_box_zero.ndim

# %%
egg_box_zero.shape

# %%
egg_box_two = np.array(
    [
        ["chicken_egg", "country_egg", "quail_egg"],
        ["chicken_egg", "country_egg", "quail_egg"],
    ]
)

# %%
egg_box_two

# %%
egg_box_3 = np.array(
    [
        [
            ["chicken_egg", "country_egg", "quail_egg"],
            ["chicken_egg", "country_egg", "quail_egg"],
        ],
        [
            ["chicken_egg", "country_egg", "quail_egg"],
            ["chicken_egg", "country_egg", "quail_egg"],
        ],
    ]
)

# %%
egg_box_3

# %%
egg_box_two.ndim

# %%
egg_box_two.shape

# %%
egg_box_3.shape

# ======================================================================
# # Array operations (statistics methods, arithmetic, logical, expo, sqrt, sine)
# ======================================================================

# %%
a = np.array([1, 2, 3])
a ^ 2

# %%
np.std(a)

# %%
array = np.array([1, 2, 3])


# Compute the statistical values
mean = np.mean(array)
median = np.median(array)
std_dev = np.std(array)
variance = np.var(array)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# %%
matrix = np.random.randint(1, 11, size=(3, 3))
print("Original matrix:")
print(matrix)

# Compute the determinant
determinant = np.linalg.det(matrix)
print("Determinant:", determinant)

# Compute the inverse
inverse = np.linalg.inv(matrix)
print("Inverse:")
print(inverse)

# %%
array1 = np.random.randint(1, 11, size=(3, 4))
array2 = np.random.randint(1, 11, size=(3, 4))
print("Array 1:")
print(array1)
print("Array 2:")
print(array2)

# Perform element-wise operations
addition = array1 + array2
subtraction = array1 - array2
multiplication = array1 * array2
division = array1 / array2

print("Element-wise addition:")
print(addition)
print("Element-wise subtraction:")
print(subtraction)
print("Element-wise multiplication:")
print(multiplication)
print("Element-wise division:")
print(division)

# ======================================================================
# # Array modification
# ======================================================================

# %%
egg_box

# %%
egg_box_two

# ======================================================================
# ### numpy.insert(arr, obj, values, axis=None)
# ======================================================================

# %%
np.insert(egg_box, 1, "duck_egg")

# %%
np.insert(egg_box_two, 1, "duck_egg", axis=1)

# %%
np.insert(egg_box_two, 2, "dinosaur_egg")

# %%
np.insert(egg_box_two, 1, "dino_egg", axis=0)

# ======================================================================
# ### numpy.delete(arr, obj, axis=None)[source]
# ======================================================================

# %%
egg_box_copy = np.delete(egg_box, 2)

# %%
egg_box

# %%
egg_box

# %%
egg_box_copy

# ======================================================================
# ### numpy.append(arr, values, axis=None)
# ======================================================================

# %%
egg_box

# %%
egg_box_two

# %%
np.append(egg_box, "duck_egg")

# ======================================================================
# # Access array elements (Indexing, slicing)
# ======================================================================

# %%
egg_box

# %%
egg_box_two

# %%
egg_box_3

# %%
egg_box[-3]

# %%
egg_box_two[1, 1]

# %%
egg_box_3[1, 1, 2]

# ======================================================================
# ## array[start : stop: step]
# ======================================================================

# %%
egg_box

# %%
egg_box_two

# %%
egg_box[0:10:2]

# %%
egg_box_two[0:1]

# ======================================================================
# # Array reshape
# ======================================================================

# ======================================================================
# ### 1d to 2d
# ======================================================================

# %%
egg_box = np.append(egg_box, "duck_egg")

# %%
egg_box.shape

# %%
egg_box.reshape(2, 2)

# ======================================================================
# ### 1d to 3d
# ======================================================================

# %%
# Ensure the array has 8 elements
egg_box = np.append(egg_box, ["extra_egg1", "extra_egg2", "extra_egg3", "extra_egg4"])

# Reshape the array
egg_box = egg_box.reshape(2, 2, 2)
egg_box

# %%
egg_box = np.array(
    [
        "chicken_egg",
        "country_egg",
        "quail_egg",
        "duck_egg",
        "chicken_egg",
        "country_egg",
        "quail_egg",
        "duck_egg",
    ]
)

# ======================================================================
# # Array iteration
# ======================================================================

# %%
egg_box

# %%
egg_box_two

# %%
for i in egg_box:
    print("I am ", i)

# %%
for i in egg_box_two:
    for j in i:
        print("I am ", j)

# %%
for i in np.nditer(egg_box_two):
    print("I am ", i)

# %%
for idx, i in np.ndenumerate(egg_box_two):
    print("index", idx)
    print("I am ", i)

# ======================================================================
# # Array joining
# ======================================================================

# %%
egg_box_two

# %%
egg_box_new = np.array([["egg1", "egg2", "egg3"], ["egg4", "egg5", "egg6"]])

# %%
egg_box_new

# ======================================================================
# ### hstack - row
# ======================================================================

# %%
np.hstack((egg_box_two, egg_box_new))

# ======================================================================
# ### vstack - column
# ======================================================================

# %%
np.vstack((egg_box_two, egg_box_new))

# ======================================================================
# ### dstack
# ======================================================================

# %%
np.dstack((egg_box_two, egg_box_new))

# ======================================================================
# # Array splitting
# ======================================================================

# %%
egg_box

# %%
np.array_split(egg_box, 4)

# ======================================================================
# #### hsplit(), vsplit(), dsplit() are also there
# ======================================================================

# ======================================================================
# # Copy vs view
# ======================================================================

# %%
egg_box_copy = egg_box.copy()
egg_box_view = egg_box.view()

# %%
egg_box_copy

# %%
egg_box

# %%
egg_box_view[0] = "duck_egg"

# %%
egg_box_view

# %%
egg_box

# %%
egg_box_copy[1] = "dino_egg"

# %%
egg_box

# %%
my_list = [1, 2, 3, 4, 5]
my_list + 1

# %%
my_array = np.array([1, 2, 3, 4, 5])
my_array + 1

# %%


