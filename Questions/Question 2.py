import matplotlib.pyplot as pyplot
import pandas
# since we only need one column, only take this column from the dataframe
ViolentCrimesPerPop = pandas.read_csv("../Provided Files/crime1.csv")["ViolentCrimesPerPop"]

# create the histogram
pyplot.hist(ViolentCrimesPerPop, color="tab:blue", edgecolor="blue")
print("Exit the histogram to open the box plot.")
pyplot.show()

# then the boxplot
pyplot.boxplot(ViolentCrimesPerPop)
pyplot.show()

'''
The histogram shows that the data has a slight right skew.
This is indicated by the "bulk" of the data being in the left side
of the histogram while the data almost trails off to the right.
The boxplot also indicates a right skew, due to the fact that
the bar for the median is
off-center and closer to the smaller values. The median being closer to the smaller values
indicates that there are more extreme values that are in the 3rd-4th quantiles.
Since the minimum and maximum values
from question 1 appear within the box plot as opposed to being dots outside
of it, there does not seem to be any outliers in this data. 
'''


