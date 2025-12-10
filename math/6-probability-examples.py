# Converted from 6-probability-examples.ipynb

# ======================================================================
# ## Probability in Python
# ======================================================================

# ======================================================================
# Probability tells us how likely something to happen!.Think of it as a number between 0 (never happens) and 1 (always happens). Essential for understanding uncertainty and randomness in data! 
# ======================================================================

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ======================================================================
# ## What is Probability?
# ======================================================================

# ======================================================================
# Probability measures how likely an event is to occur.
# 
# - Probability = 0 means the event never happens (impossible)
# - Probability = 1 means the event always happens (certain)
# - Probability = 0.5 means the event happens half the time (50% chance)
# 
# Formula: P(event) = (Number of ways event can happen) / (Total possible outcomes)
# ======================================================================

# %%
# Example: Probability of flipping a coin and getting heads
# A coin has 2 sides: heads (H) or tails (T)
# Getting heads is 1 way out of 2 possible outcomes

total_outcomes = 2  # Heads or Tails
heads_outcomes = 1  # Just heads

probability_heads = heads_outcomes / total_outcomes

print("Coin flip example:")
print(f"Total possible outcomes: {total_outcomes} (H or T)")
print(f"Ways to get heads: {heads_outcomes}")
print(f"Probability of heads: {heads_outcomes}/{total_outcomes} = {probability_heads}")
print()

# Example: Probability of rolling a 6 on a die
# A die has 6 sides: 1, 2, 3, 4, 5, 6
total_outcomes_die = 6
six_outcomes = 1  # Just the number 6

probability_six = six_outcomes / total_outcomes_die

print("Die roll example:")
print(f"Total possible outcomes: {total_outcomes_die} (1, 2, 3, 4, 5, 6)")
print(f"Ways to get 6: {six_outcomes}")
print(f"Probability of 6: {six_outcomes}/{total_outcomes_die} = {probability_six:.3f}")
print()

# Probability ranges
print("Probability ranges:")
print(f"Impossible event: P = 0")
print(f"Coin flip (heads): P = {probability_heads}")
print(f"Rolling 6: P = {probability_six:.3f}")
print(f"Certain event: P = 1")

# ======================================================================
# ## Simulating Coin Flips
# ======================================================================

# ======================================================================
# We can simulate random events using computers! Let's flip a coin many times and see how often we get heads.
# ======================================================================

# %%
# Simulate coin flips
np.random.seed(42)  # For reproducible results

# Flip a coin 10 times
flips_10 = np.random.choice(['H', 'T'], size=10)
heads_count_10 = np.sum(flips_10 == 'H')
probability_heads_10 = heads_count_10 / 10

print("10 coin flips:")
print(f"Results: {flips_10}")
print(f"Heads: {heads_count_10}, Tails: {10 - heads_count_10}")
print(f"Probability of heads (experimental): {probability_heads_10:.2f}")
print(f"Theoretical probability: 0.5")
print()

# Flip a coin 100 times
flips_100 = np.random.choice(['H', 'T'], size=100)
heads_count_100 = np.sum(flips_100 == 'H')
probability_heads_100 = heads_count_100 / 100

print("100 coin flips:")
print(f"Heads: {heads_count_100}, Tails: {100 - heads_count_100}")
print(f"Probability of heads (experimental): {probability_heads_100:.2f}")
print(f"Theoretical probability: 0.5")
print()

# Flip a coin 10000 times (more accurate!)
flips_10000 = np.random.choice(['H', 'T'], size=10000)
heads_count_10000 = np.sum(flips_10000 == 'H')
probability_heads_10000 = heads_count_10000 / 10000

print("10,000 coin flips:")
print(f"Heads: {heads_count_10000}, Tails: {10000 - heads_count_10000}")
print(f"Probability of heads (experimental): {probability_heads_10000:.4f}")
print(f"Theoretical probability: 0.5000")
print()
print("Notice: More flips → closer to theoretical probability of 0.5!")

# ======================================================================
# ## Rolling Dice
# ======================================================================

# ======================================================================
# Let's roll a die and see the probability of different outcomes! A fair die has 6 sides, each equally likely (probability = 1/6).
# ======================================================================

# %%
# Roll a die many times
np.random.seed(42)

# Theoretical probabilities
die_faces = [1, 2, 3, 4, 5, 6]
theoretical_prob = 1/6

print("Theoretical probability for each face:")
for face in die_faces:
    print(f"P(rolling {face}) = {theoretical_prob:.4f}")
print()

# Simulate 1000 die rolls
rolls = np.random.choice(die_faces, size=1000)

# Count occurrences
face_counts = Counter(rolls)
total_rolls = len(rolls)

print(f"Experimental results from {total_rolls} rolls:")
for face in sorted(face_counts.keys()):
    count = face_counts[face]
    experimental_prob = count / total_rolls
    print(f"Face {face}: {count} times, P = {experimental_prob:.4f} (theoretical: {theoretical_prob:.4f})")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of counts
faces_sorted = sorted(face_counts.keys())
counts_sorted = [face_counts[f] for f in faces_sorted]
ax1.bar(faces_sorted, counts_sorted, color='skyblue', alpha=0.7)
ax1.set_xlabel('Die Face')
ax1.set_ylabel('Count')
ax1.set_title('Die Roll Results (1000 rolls)')
ax1.set_xticks(faces_sorted)
ax1.grid(True, alpha=0.3)

# Bar chart of probabilities
experimental_probs = [face_counts[f] / total_rolls for f in faces_sorted]
theoretical_probs = [theoretical_prob] * 6

x = np.arange(len(faces_sorted))
width = 0.35

ax2.bar(x - width/2, experimental_probs, width, label='Experimental', alpha=0.7)
ax2.bar(x + width/2, theoretical_probs, width, label='Theoretical', alpha=0.7)
ax2.set_xlabel('Die Face')
ax2.set_ylabel('Probability')
ax2.set_title('Experimental vs Theoretical Probability')
ax2.set_xticks(x)
ax2.set_xticklabels(faces_sorted)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=theoretical_prob, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ======================================================================
# ##  Multiple Events: AND and OR
# ======================================================================

# ======================================================================
# When we combine events, we use different rules:
# 
# - **AND:** Both events happen (multiply probabilities)
# - **OR:** At least one event happens (add probabilities for mutually exclusive events)
# ======================================================================

# %%
# Example 1: Rolling two dice
# P(roll 1 AND roll 6) = P(roll 1) × P(roll 6)
# P(roll 1 OR roll 6) = P(roll 1) + P(roll 6) - P(roll 1 AND roll 6)

p_roll_1 = 1/6
p_roll_6 = 1/6

# AND rule: multiply (if independent)
p_1_and_6 = p_roll_1 * p_roll_6

# OR rule: add, but subtract intersection (for independent events)
# For mutually exclusive events: P(A or B) = P(A) + P(B)
p_1_or_6 = p_roll_1 + p_roll_6 - p_1_and_6

print("Rolling two dice:")
print(f"P(roll 1) = {p_roll_1:.4f}")
print(f"P(roll 6) = {p_roll_6:.4f}")
print()
print(f"P(roll 1 AND roll 6) = {p_roll_1:.4f} × {p_roll_6:.4f} = {p_1_and_6:.4f}")
print(f"P(roll 1 OR roll 6) = {p_roll_1:.4f} + {p_roll_6:.4f} - {p_1_and_6:.4f} = {p_1_or_6:.4f}")
print()

# Example 2: Two coin flips
# P(heads on first flip AND heads on second flip)
p_heads = 0.5
p_two_heads = p_heads * p_heads

# P(at least one heads in two flips)
# This is: 1 - P(no heads) = 1 - P(tails AND tails)
p_tails = 0.5
p_two_tails = p_tails * p_tails
p_at_least_one_heads = 1 - p_two_tails

print("Two coin flips:")
print(f"P(heads on flip 1) = {p_heads:.2f}")
print(f"P(heads on flip 2) = {p_heads:.2f}")
print()
print(f"P(both heads) = {p_heads:.2f} × {p_heads:.2f} = {p_two_heads:.2f}")
print(f"P(both tails) = {p_tails:.2f} × {p_tails:.2f} = {p_two_tails:.2f}")
print(f"P(at least one heads) = 1 - {p_two_tails:.2f} = {p_at_least_one_heads:.2f}")

# Verify with simulation
np.random.seed(42)
two_flips = np.random.choice(['H', 'T'], size=(10000, 2))
both_heads = np.sum(np.all(two_flips == 'H', axis=1))
at_least_one_heads = np.sum(np.any(two_flips == 'H', axis=1))

print()
print("Simulation results (10,000 pairs of flips):")
print(f"Both heads: {both_heads}/10000 = {both_heads/10000:.4f} (theoretical: {p_two_heads:.4f})")
print(f"At least one heads: {at_least_one_heads}/10000 = {at_least_one_heads/10000:.4f} (theoretical: {p_at_least_one_heads:.4f})")

# ======================================================================
# ## Probability Distributions
# ======================================================================

# ======================================================================
# A probability distribution shows the probability of each possible outcome. Let's visualize the probability distribution of rolling two dice and summing them!
# ======================================================================

# %%
# Sum of two dice
# Possible sums: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
# Let's calculate the theoretical probabilities

def theoretical_dice_sum():
    """Calculate theoretical probability distribution for sum of two dice"""
    sums = {}
    total_outcomes = 36  # 6 × 6
    
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            s = die1 + die2
            sums[s] = sums.get(s, 0) + 1
    
    # Convert counts to probabilities
    probabilities = {s: count/total_outcomes for s, count in sums.items()}
    return probabilities

# Calculate theoretical probabilities
theoretical_probs = theoretical_dice_sum()

print("Theoretical probability distribution (sum of two dice):")
print("Sum | Count | Probability")
print("-" * 30)
for s in sorted(theoretical_probs.keys()):
    count = int(theoretical_probs[s] * 36)
    prob = theoretical_probs[s]
    print(f"{s:3d} | {count:5d} | {prob:.4f}")

# Simulate
np.random.seed(42)
dice1 = np.random.randint(1, 7, size=10000)
dice2 = np.random.randint(1, 7, size=10000)
sums_sim = dice1 + dice2

# Count experimental
sum_counts = Counter(sums_sim)
experimental_probs = {s: count/10000 for s, count in sum_counts.items()}

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sums_sorted = sorted(theoretical_probs.keys())
theo_probs_sorted = [theoretical_probs[s] for s in sums_sorted]
exp_probs_sorted = [experimental_probs.get(s, 0) for s in sums_sorted]

# Theoretical distribution
ax1.bar(sums_sorted, theo_probs_sorted, color='skyblue', alpha=0.7)
ax1.set_xlabel('Sum of Two Dice')
ax1.set_ylabel('Probability')
ax1.set_title('Theoretical Probability Distribution')
ax1.set_xticks(sums_sorted)
ax1.grid(True, alpha=0.3)

# Comparison
x = np.arange(len(sums_sorted))
width = 0.35
ax2.bar(x - width/2, theo_probs_sorted, width, label='Theoretical', alpha=0.7)
ax2.bar(x + width/2, exp_probs_sorted, width, label='Experimental (10k rolls)', alpha=0.7)
ax2.set_xlabel('Sum of Two Dice')
ax2.set_ylabel('Probability')
ax2.set_title('Theoretical vs Experimental')
ax2.set_xticks(x)
ax2.set_xticklabels(sums_sorted)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print()
print(f"Most likely sum: {max(theoretical_probs, key=theoretical_probs.get)} (probability = {max(theoretical_probs.values()):.4f})")

# ======================================================================
# ## Expected Value
# ======================================================================

# ======================================================================
# Expected value is the average outcome if we repeat an experiment many times. Formula: E[X] = Σ(x × P(x)) for all possible outcomes x
# ======================================================================

# %%
# Example 1: Expected value of a single die roll
die_faces = [1, 2, 3, 4, 5, 6]
prob_each = 1/6

expected_value_die = sum(face * prob_each for face in die_faces)

print("Expected value of a single die roll:")
print("E[X] = Σ(x × P(x))")
print("     = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)")
print("     = (1 + 2 + 3 + 4 + 5 + 6) / 6")
print(f"     = {sum(die_faces)} / 6")
print(f"     = {expected_value_die:.2f}")
print()

# Verify with simulation
np.random.seed(42)
rolls = np.random.randint(1, 7, size=100000)
average_roll = np.mean(rolls)
print(f"Simulation average (100,000 rolls): {average_roll:.4f}")
print()

# Example 2: Expected value of sum of two dice
# Using our probability distribution from before
theoretical_probs = theoretical_dice_sum()

expected_sum = sum(s * p for s, p in theoretical_probs.items())

print("Expected value of sum of two dice:")
print("E[sum] = Σ(sum × P(sum))")
calculation = " + ".join([f"{s}×{p:.4f}" for s, p in sorted(theoretical_probs.items())])
print(f"       = {calculation}")
print(f"       = {expected_sum:.4f}")

# Verify
np.random.seed(42)
dice1 = np.random.randint(1, 7, size=100000)
dice2 = np.random.randint(1, 7, size=100000)
sums = dice1 + dice2
average_sum = np.mean(sums)

print()
print(f"Simulation average (100,000 pairs): {average_sum:.4f}")
print()
print("Notice: Expected value = average outcome over many trials!")

# ======================================================================
# ## Conditional Probability
# ======================================================================

# ======================================================================
# Conditional probability: P(A | B) = probability of A given that B happened. Formula: P(A | B) = P(A AND B) / P(B)
# ======================================================================

# %%
# Example: Rolling a die
# What's P(roll 6 | roll even number)?
# Even numbers: 2, 4, 6

# P(roll even) = 3/6 = 0.5
# P(roll 6 AND even) = 1/6 (since 6 is even)
# P(roll 6 | even) = P(6 AND even) / P(even) = (1/6) / (3/6) = 1/3

p_even = 3/6
p_6_and_even = 1/6  # 6 is an even number
p_6_given_even = p_6_and_even / p_even

print("Conditional probability example:")
print("What's P(roll 6 | roll even number)?")
print()
print("Given: We rolled an even number (2, 4, or 6)")
print("What's the probability it's a 6?")
print()
print(f"P(roll even) = {p_even:.2f}")
print(f"P(roll 6 AND even) = {p_6_and_even:.4f}")
print()
print("P(roll 6 | even) = P(6 AND even) / P(even)")
print(f"                 = {p_6_and_even:.4f} / {p_even:.2f}")
print(f"                 = {p_6_given_even:.4f}")
print()
print("Intuition: Of the 3 even numbers (2, 4, 6), one is 6.")
print(f"So P(6 | even) = 1/3 = {p_6_given_even:.4f}")

# Simulate to verify
np.random.seed(42)
rolls = np.random.randint(1, 7, size=100000)
even_rolls = rolls[rolls % 2 == 0]  # Filter to only even numbers
six_given_even = np.sum(even_rolls == 6) / len(even_rolls)

print()
print(f"Simulation: P(6 | even) ≈ {six_given_even:.4f} (theoretical: {p_6_given_even:.4f})")

# ======================================================================
# ## Visualizing Probability Concepts
# ======================================================================

# ======================================================================
# Let's create visualizations to understand probability better!
# ======================================================================

# %%
# Visualize probability convergence
# Show how experimental probability approaches theoretical as we increase sample size

np.random.seed(42)
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
experimental_probs = []

theoretical_prob = 0.5  # Probability of heads

for n in sample_sizes:
    flips = np.random.choice(['H', 'T'], size=n)
    heads_count = np.sum(flips == 'H')
    exp_prob = heads_count / n
    experimental_probs.append(exp_prob)

plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, experimental_probs, 'bo-', linewidth=2, markersize=8, label='Experimental Probability')
plt.axhline(y=theoretical_prob, color='r', linestyle='--', linewidth=2, label='Theoretical Probability (0.5)')
plt.xlabel('Number of Coin Flips')
plt.ylabel('Probability of Heads')
plt.title('Law of Large Numbers: Experimental Probability Converges to Theoretical')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.show()

print("Notice: As the number of trials increases, experimental probability")
print("        gets closer and closer to the theoretical probability!")
print()
print("This is called the Law of Large Numbers.")

