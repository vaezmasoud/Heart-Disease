import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv('https://github.com/vaezmasoud/Heart-Disease/blob/main/heart_cleveland.csv')
# Separate features and labels
X = df.drop('condition', axis=1)
y = df['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Predict probabilities for test set
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('AUC:', auc)

# Predict the probability for a new sample
new_sample = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prob = model.predict_proba(new_sample)
print('Probability of heart disease:', prob[0][1])

#Calculate fpr and tpr for different threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
#------------------------------------------------------
# Algorithm: MLPClassifier (Multi-Layer Perceptron) - Multi-Layer Neural Networks
# Accuracy: 0.5833333333333334
# AUC: 0.7723214285714286
