import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset
data = pd.read_csv('../dataset/MalwareDataset.csv')

# Dropping the 'hash' column
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

# Reshaping the data for the RNN
# RNN expects input in 3D array format [samples, timesteps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Define the RNN model
model = models.Sequential()
model.add(layers.SimpleRNN(64, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
model.add(layers.SimpleRNN(32, return_sequences=False))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5)

# Calculating the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
far = fp / (fp + tn)

# Print the metrics
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1: {:.2f}".format(f1))
print("False Alarm Rate: {:.2f}".format(far))
