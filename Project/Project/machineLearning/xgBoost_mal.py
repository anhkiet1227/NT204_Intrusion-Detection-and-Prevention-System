from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from xgboost import XGBClassifier

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

# Initialize the XGBoost Classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the classifier
xgb_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_clf.predict(X_test)

# Calculate metrics for XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# Calculating False Alarm Rate (FAR) for XGBoost
tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_test, y_pred_xgb).ravel()
far_xgb = fp_xgb / (fp_xgb + tn_xgb)

# Print the metrics for XGBoost
print("Accuracy: {:.2f}".format(accuracy_xgb))
print("Precision: {:.2f}".format(precision_xgb))
print("Recall: {:.2f}".format(recall_xgb))
print("F1: {:.2f}".format(f1_xgb))
print("False Alarm Rate: {:.2f}".format(far_xgb))