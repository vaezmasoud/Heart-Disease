import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

# Load the dataset
df = pd.read_csv('https://github.com/vaezmasoud/Heart-Disease/blob/main/heart_cleveland.csv')
# Separate features and labels
X = df.drop('condition', axis=1)
y = df['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Predict probabilities for test set
y_pred_proba = model.predict_proba(X_test)[:,1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print('AUC:', auc)

# Predict the probability for a new sample
new_sample = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prob = model.predict_proba(new_sample)
print('Probability of heart disease:', prob[0][1])
if prob[0][1]>= 0.5:
  print("condition: disease")
else:
  print("condition: no disease")
#------------------------------------------------------
# Model: Gaussian Naive Bayes
# Accuracy: 0.7666666666666667
# AUC: 0.8415178571428572
