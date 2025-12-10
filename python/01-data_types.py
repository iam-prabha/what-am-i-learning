# Converted from 01-data_types.ipynb

# %%
item_id = 1

# %%
item_name = "rice"

# %%
item_quantity = 2.5

# %%
needs_restock = False

# %%
type(item_id)

# %%
print(item_id)

# %%
type(item_name)

# %%
type(item_quantity)

# %%
type(needs_restock)

# ======================================================================
# # checkpoints
# ======================================================================

# ======================================================================
# - variable name can only start with '_' or 'str'
# - should be descriptive
# - case sensitive
# ======================================================================

# %%
_item_no = 12

# %%
item_no = 12345

# %%
ITEM_NO = 12345

# %%
item12_no = 123456


# ======================================================================
# # Variable declaration
# ======================================================================

# %%
item_id = int(2)
item_name = str("paruppu")
item_quantity = float(1)
needs_restock = bool(True)

# %%
type(needs_restock)

# %%
a = 1
b = 1

type(a == b)

# ======================================================================
# # Dynamic typing
# ======================================================================

# %%
print(item_name)

# ======================================================================
# # Type conversion
# ======================================================================

# ======================================================================
# - int to str
# - int to float
# - int to bool
# - but not str to int
# - boolean 0 = false, all others boolean = True
# ======================================================================

# %%
type(item_quantity)

# %%
int(item_quantity)

# %%
type(int(item_quantity))

# %%
bool(9)

# %%
bool(item_name)
