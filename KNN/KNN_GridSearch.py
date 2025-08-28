# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import numpy as np

# Step 1: Load the data from the CSV file
data = pd.read_csv('Cleaned_BigMartWaste.csv')

# Step 2: Select the features to be used
selected_features = ['Weight', 'FatContent', 'MRP']
X = data[selected_features]  # Using only the selected features
y = data['ProductType']

# Step 3: Split the data into a training+validation set (80%) and a test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the KNN model with GridSearchCV and Pipeline
pipeline = Pipeline([
   ('scaler', StandardScaler()),  # Scale features
   ('knn', KNeighborsClassifier())  # KNN classifier
])

# Step 5: Define the parameter grid for GridSearchCV
param_grid = {
   'knn__n_neighbors': [3, 5, 7, 9],  # Number of neighbors
   'knn__weights': ['uniform', 'distance'],  # Weighting function
   'knn__metric': ['euclidean', 'manhattan'],  # Distance metric
}

# Step 6: Perform GridSearchCV for hyperparameter tuning with cross-validation (on training+validation set)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train_val, y_train_val)

# Step 7: Display the best parameters and best score from GridSearchCV
print("Best parameters from Grid Search:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Step 8: Perform K-Fold Cross-Validation on the training+validation set
best_model = grid_search.best_estimator_  # Extract the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # K-Fold Cross-Validation with 5 folds
scores = []  # List to store accuracy scores for each fold
all_y_true = []  # List to store true labels for all folds
all_y_pred = []  # List to store predicted labels for all folds

print(f"\nPerforming {kf.get_n_splits()}-Fold Cross-Validation on the training+validation set:")

# Loop through each fold
for i, (train_index, val_index) in enumerate(kf.split(X_train_val)):
   X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
   y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

   # Fit the model on the current fold
   best_model.fit(X_train, y_train)

   # Predict on the validation set
   y_pred_val = best_model.predict(X_val)

   # Store true and predicted labels for later metrics calculation
   all_y_true.extend(y_val)
   all_y_pred.extend(y_pred_val)

   # Calculate accuracy, precision, recall, and F1-score on the validation set
   accuracy = accuracy_score(y_val, y_pred_val)
   precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
   recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
   f1 = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

   # Append accuracy to the scores list
   scores.append(accuracy)

   # Print results for this fold
   print(f"\nFold {i + 1} Results (Validation Set):")
   print(f"Accuracy: {accuracy:.2f}")
   print(f"Precision: {precision:.2f}")
   print(f"Recall: {recall:.2f}")
   print(f"F1-Score: {f1:.2f}")
   print("Classification Report:")
   print(classification_report(y_val, y_pred_val))

# Step 9: Calculate and display overall performance from K-Fold Cross-Validation
print("\nOverall Cross-Validation Results on Validation Set:")
print(f"Mean Accuracy: {np.mean(scores):.2f}")
print(f"Standard Deviation of Accuracy: {np.std(scores):.2f}")

# Step 10: Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Performance metrics on the test set
test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print("\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1-Score: {test_f1:.2f}")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix on Test Set:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC-ROC
y_prob_test = best_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test, pos_label=best_model.classes_[1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
