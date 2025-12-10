# Converted from 08-matplotlib.ipynb

# ======================================================================
# # Libraries
# ======================================================================

# %%
import matplotlib.pyplot as plt

# %%
import pandas as pd

# ======================================================================
# # Load data
# ======================================================================

# %%
cyclones = pd.read_csv("../data/chennai_cyclones.csv")

# %%
cyclones.info()

# %%
cyclones

# ======================================================================
# # Basic line plot
# ======================================================================

# %%
plt.plot(cyclones["Year"], cyclones["Fatalities"])
plt.xlabel("Year")
plt.ylabel("death rate")
plt.title("Chennai floods")
plt.show()

# ======================================================================
# # Plot beautifiers (colour, marker, linewidth, grid, linestyle, markersize)
# ======================================================================

# %%
plt.plot(
    cyclones["Year"],
    cyclones["Fatalities"],
    color="red",
    marker="o",
    linewidth=5,
    linestyle="dotted",
    markersize="15",
)
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("death rate")
plt.title("Chennai floods")
plt.show()

# ======================================================================
# # Multiple plots
# ======================================================================

# %%
fig = plt.figure(figsize=(25, 15))  # width , height

ax1 = fig.add_subplot(2, 2, 1)

plt.plot(
    cyclones["Year"],
    cyclones["Fatalities"],
    color="red",
    marker="o",
    linewidth=5,
    linestyle="dotted",
    markersize="15",
)
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("death rate")
plt.title("Chennai floods")

ax2 = fig.add_subplot(2, 2, 2)

plt.plot(
    cyclones["Cyclone Name"],
    cyclones["Fatalities"],
    color="blue",
    marker="o",
    linewidth=5,
    linestyle="dotted",
    markersize="15",
)
plt.grid(True)
plt.xlabel("name of cyclone")
plt.ylabel("death rate")

ax3 = fig.add_subplot(2, 2, 3)

plt.plot(
    cyclones["Formed"],
    cyclones["Fatalities"],
    color="hotpink",
    marker="o",
    linewidth=5,
    linestyle="dotted",
    markersize="15",
)
plt.grid(True)
plt.xlabel("cyclone birth")
plt.ylabel("death rate")

ax4 = fig.add_subplot(2, 2, 4)

plt.plot(
    cyclones["Dissipated"],
    cyclones["Fatalities"],
    color="purple",
    marker="o",
    linewidth=5,
    linestyle="dotted",
    markersize="15",
)
plt.grid(True)
plt.xlabel("cyclone death")
plt.ylabel("death rate")

# ======================================================================
# # Bar plot
# ======================================================================

# %%
def convert_damages(damages):
    endresult = []
    for i in damages:
        values = i.split()
        # print(values)
        num = values[0][1:]
        print(type(num))
        if values[1] == "million":
            result = float(num) * 1000000
        else:
            result = float(num) * 1000000000
        endresult.append(result)
    return endresult

# %%
damage_converted = convert_damages(cyclones["Damage"])

# %%
damage_converted

# %%
cyclones["Damage_converted"] = damage_converted

# %%
cyclones

# %%
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)

ax.xaxis.label.set_color("Green")  # setting up X-axis label color to yellow
ax.yaxis.label.set_color("blue")  # setting up Y-axis label color to blue

ax.tick_params(axis="x", colors="red")  # setting up X-axis tick color to red
ax.tick_params(axis="y", colors="black")


plt.bar(cyclones["Cyclone Name"], cyclones["Damage_converted"])
plt.xlabel("name of cyclone")
plt.xticks(rotation=45)
plt.ylabel("damages")
plt.title("which cyclone is expensive?")
plt.show()

# ======================================================================
# # Histogram
# ======================================================================

# %%
plt.hist(cyclones["Year"], bins=5, edgecolor="black")
plt.show()

# ======================================================================
# # Scatterplot
# ======================================================================

# %%
cyclones["cyclone_birth_month"] = cyclones["Formed"].apply(lambda x: x.split()[0])

# %%
cyclones

# %%
plt.scatter(cyclones["Cyclone Name"], cyclones["cyclone_birth_month"])
plt.xticks(rotation=45)
plt.show()

# %%
cyclones["Formed"]

# ======================================================================
# # Pie chart
# ======================================================================

# %%
cyclones

# %%
cyclones["death_percentage"] = (
    cyclones["Fatalities"] / cyclones["Fatalities"].sum() * 100
)

# %%
plt.figure(figsize=(15, 10))
plt.pie(cyclones["death_percentage"], labels=cyclones["Cyclone Name"], shadow=True)

# %%


# %%


