from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Importing the dataset
data = pd.read_csv('./MalwareDataset.csv')

# Dropping the 'hash' column as it's not useful for prediction
data = data.drop(columns=['hash'])

# Convert categorical data to numerical data
data['classification'] = data['classification'].map({'malware': 1, 'benign': 0})

# Separate the features and the target variable
X = data.drop('classification', axis=1)
y = data['classification']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier()

# Train the classifier
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test)

# Calculate metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Calculating False Alarm Rate (FAR) for Random Forest
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_test, y_pred_rf).ravel()
far_rf = fp_rf / (fp_rf + tn_rf)

# Print the metrics for Random Forest
print("Accuracy: {:.2f}".format(accuracy_rf))
print("Precision: {:.2f}".format(precision_rf))
print("Recall: {:.2f}".format(recall_rf))
print("F1: {:.2f}".format(f1_rf))
print("False Alarm Rate: {:.2f}".format(far_rf))