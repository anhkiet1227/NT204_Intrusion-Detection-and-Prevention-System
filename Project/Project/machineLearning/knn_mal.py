import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data = pd.read_csv('../dataset/MalwareDataset.csv')  # Update the file path

# Dropping the 'hash' column as it's not useful for prediction
data = data.drop(columns=['hash'])

# Convert categorical data to numerical data
data['classification'] = data['classification'].map({'malware': 1, 'benign': 0})

# Separate the features and the target variable
X = data.drop('classification', axis=1)
y = data['classification']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)  # You can experiment with different values of n_neighbors

# Train the classifier
knn_clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn_clf.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
far = fp / (fp + tn)  # False Alarm Rate

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"False Alarm Rate: {far:.2f}")
