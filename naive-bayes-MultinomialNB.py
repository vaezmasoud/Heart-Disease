import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
df = pd.read_csv('https://github.com/vaezmasoud/Heart-Disease/blob/main/heart_cleveland.csv')
# Separate features and labels

# Separate features and labels
X = df.drop('condition', axis=1)
y = df['condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Predict the probabilities for the test set
y_probs = model.predict_proba(X_test)

# Calculate the AUC score
auc = roc_auc_score(y_test, y_probs[:,1])

# Print the AUC score
print('AUC score:', auc)

# Predict the probability for a new sample
new_sample = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prob = model.predict_proba(new_sample)
print('Probability of heart disease:', prob[0][1])
if prob[0][1]>= 0.5:
  print("condition: disease")
else:
  print("condition: no disease")

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:,1])

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()
#------------------------------------------------------
# Model: Multinomial Naive Bayes
# Accuracy: 0.6
#AUC: 0.7399553571428571
