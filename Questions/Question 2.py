import matplotlib.pyplot as pyplot
import pandas
# since we only need one column, only take this column from the dataframe
ViolentCrimesPerPop = pandas.read_csv("../Provided Files/crime1.csv")["ViolentCrimesPerPop"]

print(ViolentCrimesPerPop)
# create the histogram
pyplot.hist(ViolentCrimesPerPop, color="tab:blue", edgecolor="blue")

# create the histogram's title, x-axis label,
# and y-axis label
pyplot.title("Violent Crimes Per Population")

pyplot.xlabel("Proportion of Violent Crimes Per Population")
pyplot.ylabel("Amount")

print("Exit the histogram to open the box plot.")
pyplot.show()

# do the same for the boxplot
pyplot.boxplot(ViolentCrimesPerPop)
pyplot.title("Violent Crimes Per Population")
pyplot.ylabel("Proportion of Violent Crimes Per Population")
pyplot.show()

# i find thes types of comments easier to write in

"""
The histogram shows that the data has a very slight right skew.
This is indicated by the a larger amount of the data being in the left side
of the histogram while the data almost trails off to the right.

The boxplot also indicates a right skew, due to the fact that the
bar for the median is closer to the smaller values than the boxplot's center.
Since the minimum and maximum values from question 1 appear at
the end of the as box plot's whiskers opposed to being dots
outside of the plot, there does not seem to be any outliers in this data. 
"""


