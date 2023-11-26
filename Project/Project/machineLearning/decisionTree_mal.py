from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Importing the dataset
data = pd.read_csv('../dataset/MalwareDataset.csv')

# Dropping the 'hash' column as it's not useful for prediction
data = data.drop(columns=['hash'])

# Convert categorical data to numerical data
data['classification'] = data['classification'].map({'malware': 1, 'benign': 0})

# Separate the features and the target variable
X = data.drop('classification', axis=1)
y = data['classification']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculating False Alarm Rate (FAR)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
far = fp / (fp + tn)

# Print the metrics
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1: {:.2f}".format(f1))
print("False Alarm Rate: {:.2f}".format(far))