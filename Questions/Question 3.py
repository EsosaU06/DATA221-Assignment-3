from sklearn.model_selection import train_test_split

import pandas
kidneyDiseaseCSV = pandas.read_csv("../Provided Files/kidney_disease.csv")

# classification Y
kidneyDiseaseClassification_targetColumn = kidneyDiseaseCSV["classification"]

# remaining dataset
kidneyDisease_featureMatrix = kidneyDiseaseCSV.drop("classification", axis=1)

kidneyDisease_train, kidneyDisease_test, classification_train, classification_test = train_test_split(kidneyDisease_featureMatrix,
                                                                                                      kidneyDiseaseClassification_targetColumn,
                                                                                                      test_size=0.30, random_state=5)


'''
We should not train and test a model on the same data because if our testing data is
already in our training data, the model would predict data by memorization instead
of truly learning from the data.

The purpose of the testing set is to have values we can compare the model's
output to in order to evalute how accurate the model is.
'''