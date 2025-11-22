# Statistical Concepts for Beginners

Welcome to this guide on fundamental statistical concepts! This document covers key topics like **Mean, Median, Mode**, **Percentiles & Quartiles**, **Range, Variance & Standard Deviation**, **Normalization & Standardization**, **Covariance & Correlation**, **Hypothesis Testing (P-value)**, and **Z-test**. Each section includes simple explanations, formulas, examples with test scores, and real-world uses—perfect for revising later.

---

## Introduction

Statistics is all about making sense of numbers. Whether you’re figuring out class averages, spotting trends, or deciding if something’s unusual, these concepts are your toolkit. Let’s break them down step-by-step!

---

## 1. Mean, Median, Mode

- These three help you find the "center" of a dataset in different ways.

- **Mean (Average)**: Add all the values and divide by how many there are.  
- **Formula**:  Mean = (Sum of all values) / (Number of values)
  
- **Example**: Test scores: 2, 3, 7
    
- Mean = (2 + 3 + 7) / 3 = 12 / 3 = 4  

- **Use**: Shows the typical score in a class.  
- **Visual**: Think of a seesaw balancing at 4. 

2   3   7
|   |   |
----4---- (balancing point)


- **Median (Middle)**: The middle value when numbers are in order. (For an even number of values, average the two middle ones.)  
- **Example**: Scores: 2, 3, 7 → Ordered: 2, 3, 7 → Median = 3

- Scores: 2, 3, 7, 8 → Median = (3 + 7) / 2 = 5  

- **Use**: Great when extreme scores (like a 0 or 100) skew the mean.  
- **Visual**: Splits the data in half.  

2   3   7
    ↑
  Median



- **Mode (Most Common)**: The value that shows up most often.  
- **Example**: Scores: 2, 3, 3, 7 → Mode = 3 (appears twice)  
- **Use**: Tells you the most popular score.  
- **Visual**: In a bar graph, it’s the tallest bar.  

Height:  2  1
Value:   2  3  7
|  |
3 is tallest!


---

## 2. Percentiles & Quartiles

- These show where a value fits compared to the rest of the data.

- **Percentiles**: What percentage of the data is below a value.  
- **Example**: 75th percentile means you beat 75% of the class.  
- **Use**: Compare your rank, like being in the "top 25%."

- **Quartiles**: Split the data into four equal parts.  
- **Q1 (25th percentile)**: 25% below this value.  
- **Q2 (50th percentile)**: The median.  
- **Q3 (75th percentile)**: 75% below this value.  
- **Example**: Scores: 1, 3, 5, 7, 9  
- Ordered: 1, 3, 5, 7, 9  
- Q1 = 3, Q2 = 5 (median), Q3 = 7  
- **Visual**: A box plot shows it best. 

1 ---- 3 ---- 5 ---- 7 ---- 9
|    Q1    Q2    Q3    |

- **Use**: See how spread out scores are or spot outliers.

---

## 3. Range, Variance, & Standard Deviation

- These measure how spread out the data is.

- **Range**: Biggest value minus the smallest.  
- **Formula**:  Range = Max - Min

- **Example**: Scores: 2, 3, 7 → Range = 7 - 2 = 5  
- **Use**: Quick check of how much scores vary.

- **Variance**: Average of the squared differences from the mean.  
- **Formula**:  Variance = [Sum of (each value - mean)²] / (Number of values)


- **Example**: Scores: 2, 3, 7  
- Mean = 4  
- Differences: (2-4)² = 4, (3-4)² = 1, (7-4)² = 9  
- Variance = (4 + 1 + 9) / 3 = 14 / 3 ≈ 4.67  
- **Use**: Shows how much scores differ from the average.

- **Standard Deviation**: Square root of variance (easier to interpret).  
- **Formula**:  Standard Deviation = √Variance


- **Example**: √4.67 ≈ 2.16  
- **Visual**: On a bell curve, it shows the spread.  

/

/  

/    

2   4   6   (mean = 4, SD ≈ 2.16)

- **Use**: Tells you the typical distance from the mean.

---

## 4. Normalization & Standardization

- These adjust data to make comparisons easier.

- **Normalization**: Scales data to a range (usually 0 to 1).  
- **Formula**:  Normalized Value = (Value - Min) / (Max - Min)

- **Example**: Scores: 2, 3, 7  
- Min = 2, Max = 7  
- 2 → (2-2) / (7-2) = 0  
- 7 → (7-2) / (7-2) = 1  
- 3 → (3-2) / (7-2) = 0.2  
- **Use**: Compare scores from different scales (e.g., 0-10 vs. 0-100).

- **Standardization**: Makes the mean 0 and standard deviation 1.  
- **Formula**:  Standardized Value = (Value - Mean) / (Standard Deviation)

- **Example**: Scores: 2, 3, 7 (Mean = 4, SD ≈ 2.16)  
- 2 → (2-4) / 2.16 ≈ -0.93  
- 7 → (7-4) / 2.16 ≈ 1.39  
- **Use**: Compare scores across tests using Z-scores.

---

## 5. Covariance & Correlation

- These show how two variables (like scores from two classes) relate.

- **Covariance**: Do they move together? Positive = yes, negative = opposite.  
- **Formula**:  Covariance = [Sum of (x - mean_x)(y - mean_y)] / (Number of values)

- **Example**: Class A: 2, 3, 7; Class B: 4, 5, 9  
- Mean_A = 4, Mean_B = 6  
- Covariance ≈ 4 (positive, they rise together)  
- **Use**: Check if good scores in one class mean good scores in another.

- **Correlation**: Strength and direction of the relationship (-1 to 1).  
- **Formula**:  Correlation = Covariance / (SD_x × SD_y)

- **Example**: Covariance = 4, SD_A ≈ 2.16, SD_B ≈ 2.16  

- Correlation ≈ 4 / (2.16 × 2.16) ≈ 0.86 (strong positive)

- **Visual**: Scatter plot shows it. Tight line = strong correlation. 

B: 9 |      x
5 |   x
4 | x
|____
2 3 7 A


- **Use**: Predict one variable from another.

---

## 6. Hypothesis Testing (P-value)

- This helps you test ideas with data.

- **Hypothesis**: A guess to test.  
- **Null (H0)**: No change (e.g., "Class average is 70").  
- **Alternative (H1)**: There’s a change (e.g., "Not 70").

- **P-value**: Chance of seeing your data if H0 is true.  
- Small P-value (< 0.05) → Reject H0 (significant).  
- Big P-value (≥ 0.05) → Keep H0 (not significant).  
- **Example**: If P-value = 0.03, the average isn’t 70 (significant).  
- **Use**: Decide if your class average is unusual.  
- **Visual**: P-value is the tail area of a bell curve. 

/

/  

/    

----reject H0 here----


---

## 7. Z-test

- A test to compare a sample to a population when the population SD is known.

- **Formula**:  Z = (Sample Mean - Population Mean) / (Population SD / √Sample Size)

- **Example**: Population mean = 70, SD = 10, Sample (100 students) mean = 72  
- Z = (72 - 70) / (10 / √100) = 2 / 1 = 2

- P-value ≈ 0.045 (< 0.05) → Reject H0.  
- **Use**: See if your class differs from the population average.  
- **Visual**: Z = 2 is in the tail of the bell curve.  

/

/  

/    \  ← Z = 2
-1.96  1.96


---

## Conclusion

You’ve now got the basics of statistics down! From finding the center (Mean, Median, Mode) to measuring spread (Range, Variance, Standard Deviation), comparing data (Normalization, Standardization), and testing ideas (Hypothesis Testing, Z-test), you’re ready to tackle numbers. Try these concepts with real data—like your test scores or sports stats—to keep them fresh. Happy revising!

---

### Practice Exercises

Test yourself with these:

1. Find the mean, median, and mode for: 85, 90, 92, 90, 88.  
2. Calculate Q1, Q2, Q3 for: 10, 20, 30, 40, 50, 60, 70.  
3. Compute range, variance, and SD for: 5, 7, 9.  
4. Normalize: 10, 20, 30 to 0-1.  
5. Standardize: 5, 10, 15 (mean = 10, SD = 5).  
6. Correlation = 0.9. What does it mean?  
7. P-value = 0.03. What’s your decision?

---