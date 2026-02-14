import pandas

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
kidneyDiseaseCSV = pandas.read_csv("../Provided Files/kidney_disease.csv")
from sklearn.metrics import accuracy_score

# use the same training and testing data from before

kidneyDiseaseClassification_targetColumn = kidneyDiseaseCSV["classification"].fillna(0)
kidneyDiseaseClassification_targetColumn = kidneyDiseaseClassification_targetColumn.map({"ckd": 1, "notckd": 0, "ckd\t":1})

kidneyDisease_featureMatrix = kidneyDiseaseCSV.drop("classification", axis=1)

kidneyDisease_featureMatrix = kidneyDisease_featureMatrix.fillna(0)
kidneyDisease_featureMatrix = pandas.get_dummies(kidneyDisease_featureMatrix,dtype=int)

kidneyDisease_train, kidneyDisease_test, classification_train, classification_test = train_test_split(kidneyDisease_featureMatrix, kidneyDiseaseClassification_targetColumn, test_size=0.30, random_state=5)

# the list of k-values given
KValues = [1,3,5,7,9]

KNNAccuracyList = [] # put the resulting accuracy scores here

for KValue in KValues:
    # train the model from each K Value
    kidneyDiseaseKNNModel = KNeighborsClassifier(n_neighbors=KValue)
    kidneyDiseaseKNNModel.fit(kidneyDisease_featureMatrix, kidneyDiseaseClassification_targetColumn)
    
    #  the model then predicts the labels from the X test data
    kidneyDiseasePredictedLabels = kidneyDiseaseKNNModel.predict(kidneyDisease_test)
    
    kidneyDiseaseKNNModelAccuracy = accuracy_score(kidneyDiseasePredictedLabels, classification_test)
    KNNAccuracyList.append(kidneyDiseaseKNNModelAccuracy)

# create a table where each column corresponds to a kvalue and a result
KValueTable = pandas.DataFrame([KValues, KNNAccuracyList], index=["K-Value","Accuracy Score"])

print(KValueTable) # output:
"""
                  0         1         2         3         4
K-Value         1.0  3.000000  5.000000  7.000000  9.000000
Accuracy Score  1.0  0.991667  0.983333  0.983333  0.983333
"""
###

"""
The highest accuracy score comes from the model trained with 1 nearest neighbour.
The model with the second highest accuracy score uses 3. Then the same
accuracy scores come rom the rest of the K values chosen (5,7, and 9).
Changing the value of k affects how many nearby points (neighbours) the model uses
to predict a value. A higher K means more neighbours are used.

Very small values of K may cause overfitting because it only has one data point to learn from.
As a result, the not does not generalize the data it learns from.
While very large values of k may cause underfitting because it will take
from too many points. This will lead to overgeneralizing.
"""