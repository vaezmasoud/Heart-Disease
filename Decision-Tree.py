import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

data = pd.read_csv('/content/heart_cleveland_upload.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

y_pred_proba = classifier.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")

new_data = [[60, 1, 0, 130, 233, 1, 2, 150, 0, 2.3, 1, 0, 3]]
# [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prediction = classifier.predict(new_data)
print(f"Prediction: {prediction}")
#------------------------------------------------------
# Algorithm: Decision Tree
# Accuracy: 0.7166666666666667
# AUC: 0.7522321428571428
