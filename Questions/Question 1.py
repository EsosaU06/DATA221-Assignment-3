import pandas
crimeDataFrame = pandas.read_csv("../Provided Files/crime1.csv")

# taking the statistics from one column
violentCrimesPerPopColumn = crimeDataFrame["ViolentCrimesPerPop"]

print("Mean:",violentCrimesPerPopColumn.mean())
print("Median:",violentCrimesPerPopColumn.median())

print("Standard deviation:",violentCrimesPerPopColumn.std())

print("Minimum value:",violentCrimesPerPopColumn.min())
print("Maximum value:",violentCrimesPerPopColumn.max())

# The mean is 0.05 more than the median. This indicates that the data is somewhat skewed right.
# If there are extreme values, the mean is more affected than the median. This is because
# each value is used to calculate the mean, while the formula for the median
# does not work directly with outliers.