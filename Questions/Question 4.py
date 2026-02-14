import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

kidneyDiseaseCSV = pandas.read_csv("../Provided Files/kidney_disease.csv")

# classification Y
# clean up the data, turn every categorical value into numerical data
# (issues in cleaning up the code were found by generative AI)

# todo: consider dropping rows without data
kidneyDiseaseClassification_targetColumn = kidneyDiseaseCSV["classification"].fillna(0)
kidneyDiseaseClassification_targetColumn = kidneyDiseaseClassification_targetColumn.map({"ckd": 1, "notckd": 0, "ckd\t":1})

# remaining dataset, what the model will predict values from:
kidneyDisease_featureMatrix = kidneyDiseaseCSV.drop("classification", axis=1)

# once again, convert all the categorical data into numerical data
kidneyDisease_featureMatrix = kidneyDisease_featureMatrix.fillna(0)
kidneyDisease_featureMatrix = pandas.get_dummies(kidneyDisease_featureMatrix,dtype=int)

# split the testing and training data
kidneyDisease_train, kidneyDisease_test, classification_train, classification_test = train_test_split(kidneyDisease_featureMatrix, kidneyDiseaseClassification_targetColumn, test_size=0.30, random_state=5)

KNNModel = KNeighborsClassifier(n_neighbors = 5)
kidneyDiseaseTrainedKNNModel = KNNModel.fit(kidneyDisease_train, classification_train)

predictedYValues = kidneyDiseaseTrainedKNNModel.predict(kidneyDisease_test)

# confusion matrix:
kidneyDiseaseConfusionMatrix = metrics.confusion_matrix(classification_test, predictedYValues)

# precision, accurazy, recall, and f1 scores
kidneyDiseasePrecision = precision_score(classification_test, predictedYValues)
kidneyDiseaseAccuracy = accuracy_score(classification_test, predictedYValues)
kidneyDiseaseRecall = recall_score(classification_test, predictedYValues)
kidneyDiseaseF1 = f1_score(classification_test, predictedYValues)

print("Confusion matrix:")
print(kidneyDiseaseConfusionMatrix)
print("Precision test: ",kidneyDiseasePrecision)
print("Accuracy Test: ",kidneyDiseaseAccuracy)
print("Recall Test: ",kidneyDiseaseRecall)
print("F1 Test: ",kidneyDiseaseF1)

'''
A True Positive is when the model accurately detects kidney disease. Meaning that kidney
disease was predicted given the patient does in fact have kidney disease.
A True Negative is when the model accurately detects no kidney disease.
False Positive is when the model detects kidney disease when the patient does not have kidney disease.
False Negative is when the model detects no kidney disease when the patient has kidney disease.

Accuracy may not be enough because a high accuracy score could be attributed to the model
being overfitted. This means that the model may not be accurate in predicting values from
another set of data, even if the accuracy score is high.

When missing a kidney disease case is very important, recall is the most important metric.
This is because recall is essentially the rates of true positives. If the recall rate is high,
the chance of detecting a kidney disease case when the patient has kidney disease is also high.

'''