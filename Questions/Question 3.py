from sklearn.model_selection import train_test_split
import pandas

kidneyDiseaseCSV = pandas.read_csv("../Provided Files/kidney_disease.csv")

# the Y the model will try to predict
kidneyDiseaseClassification_targetColumn = kidneyDiseaseCSV["classification"]

# remaining dataset
kidneyDisease_featureMatrix = kidneyDiseaseCSV.drop("classification", axis=1)

# split the data into training and testing data
# 70-30 split between the training and testing data
# is achieved by setting the test size to 0.30 (30%)
kidneyDisease_train, kidneyDisease_test, classification_train, classification_test = train_test_split(kidneyDisease_featureMatrix,
                                                                                                      kidneyDiseaseClassification_targetColumn,
                                                                                                      test_size=0.30, random_state=5)


"""
We should not train and test a model on the same data because if our testing data is
already in our training data, the model would predict data by memorization instead
of learning from the data.

The purpose of the testing set is to have values we can compare the model's
output to in order to evalute how accurately the model can predict values.
"""