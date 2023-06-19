import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

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

# Predict the probability for a new sample
new_sample = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prob = model.predict_proba(new_sample)
print('Probability of heart disease:', prob[0][1])
if prob[0][1]>= 0.5:
  print("condition: disease")
else:
  print("condition: no disease")
#------------------------------------------------------
# Model: Multinomial Naive Bayes
# Accuracy: 0.6
