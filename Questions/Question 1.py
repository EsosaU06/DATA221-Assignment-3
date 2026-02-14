import pandas
crimeDataFrame = pandas.read_csv("../Provided Files/crime1.csv")

# taking the statistics from one column
violentCrimesPerPopColumn = crimeDataFrame["ViolentCrimesPerPop"]

print("Mean:",violentCrimesPerPopColumn.mean()) # 0.44119122257053295
print("Median:",violentCrimesPerPopColumn.median()) #0.39

print("Standard deviation:",violentCrimesPerPopColumn.std()) #0.2763505847811399

print("Minimum value:",violentCrimesPerPopColumn.min()) #0.02
print("Maximum value:",violentCrimesPerPopColumn.max()) #1.0

# The mean is 0.05 more than the median.
# Which indicates that the data is somewhat skewed to the right.
# The mean is more affected by outliers than the median. This is because
# each data value is used to calculate the mean, including extreme values.
# The formula for the median
# does not contain every single value, so it is less affected
# by extreme values.