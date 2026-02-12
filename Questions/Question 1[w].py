import pandas
crimeDataFrame = pandas.read_csv("../Provided Files/crime.csv")

crimeDataFrameViolentCrimesPerPopColumn = crimeDataFrame["ViolentCrimesPerPop"]

print("Mean:",crimeDataFrameViolentCrimesPerPopColumn.mean())

print("Median:",crimeDataFrameViolentCrimesPerPopColumn.median())

print("Standard deviation:",crimeDataFrameViolentCrimesPerPopColumn.std())

print("Minimum value:",crimeDataFrameViolentCrimesPerPopColumn.min())

print("Maximum value:",crimeDataFrameViolentCrimesPerPopColumn.max())
