# Converted from 02-loops_and_conditional.ipynb

# ======================================================================
# # Functions
# ======================================================================

# %%
def coconut_chutney():
    print("COCONUT CHUTNEY RECIPE:")
    print("coconut+potukkadalai + chilli + salt")

# %%
for i in range(10):
    coconut_chutney()

# %%
def tomato_chutney():
    print("tomato + onion + chilli + salt")

# %%
coconut_chutney()

# ======================================================================
# # For loop
# ======================================================================

# %%
range(7)

# %%
for i in range(7):
    print(i)

# %%
for k in range(7):
    coconut_chutney()

# %%
k = 0

# %%
for k in range(7):
    print("day", k, "- RECIPE")
    coconut_chutney()

# ======================================================================
# # while loop
# ======================================================================

# %%
day = 0
while (day < 7):
    print("day", day,"- recipe")
    coconut_chutney()
    day = day +1

# ======================================================================
# # Conditional statements
# ======================================================================

# ======================================================================
# - if
# - else
# - elif
# - continue
# - break
# - pass
# ======================================================================

# %%
coconut_availability = int(input())

# %%
coconut_availability

# %%
for i in range(7):
    print("day", i, "- recipe")
    if coconut_availability == 1:
        coconut_chutney()
    elif coconut_availability == 0:
        tomato_chutney()
    else:
        print("make sambar...")


# %%
coconut_availability = int(input())

# %%
coconut_availability_list = [1,0,0,1,1,0,0]

# %%
type(coconut_availability_list)

# %%
coconut_availability_list[3]

# %%
for i in range(7):
    print("day",i,"- recipe")
    if coconut_availability_list[i] == 1:
        coconut_chutney()
    elif coconut_availability_list[i] == 0:
        tomato_chutney()
    else:
        print("make sambar")

# ======================================================================
# # Short hand if else
# ======================================================================

# ======================================================================
# # Ternary operation or conditional expressions
# ======================================================================

# %%
coconut_chutney() if coconut_availability == 1 else tomato_chutney()

# %%
coconut_availability

# ======================================================================
# # Conditional statements inside loops
# ======================================================================

# ======================================================================
# - break : to break a loop
# - continue : to stop what we are doing immediately yet continue
# - pass: do nothing
# ======================================================================

# %%
for j in range(3):
    print("week no:", j,"--------------------------")
    for i in range(7):
        print("day",i,"- recipe")
        if coconut_availability_list[i] == 1:
            coconut_chutney()
            # continue
            # print("yeahh it is coconut chuthey today!")
        elif coconut_availability_list[i] == 0:
            # break
            # tomato_chutney()
            # pass
        else:
            print("make sambar")


# ======================================================================
# # Nested loops
# ======================================================================

# %%
for j in range(3):
    print("week no:", j,"--------------------------")
    for i in range(7):
        print("day",i,"- recipe")
        if coconut_availability_list[i] == 1:
            coconut_chutney()
        elif coconut_availability_list[i] == 0:
            tomato_chutney()
        else:
            print("make sambar")

# %%
a = 5
b= 10
c= 15

if a<b:
    if b<c:
        print("c is big")
    else:
        print("b is big")
else:
    print("a is big")

# %%
x = 5
if x>10:
    print("x is greater than 10")
elif x==5:
    print("x is 5")
else:
    print("x is just dummy")

# %%
for i in range(1, 10, 3):
    print(i)

# %%
