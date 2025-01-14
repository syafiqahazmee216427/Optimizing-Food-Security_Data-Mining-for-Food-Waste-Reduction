import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the dataset
df = pd.read_csv('Cleaned_BigMartWaste.csv')

# Step 2: Select features and target
selected_features = ['MRP', 'Weight', 'ProductVisibility']  # Selected features
X = df[selected_features]  # Features
y = df['ProductType']  # Target

# Step 3: Split the data into a training+validation set (80%) and a test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the pipeline
pipeline = Pipeline([
   ('scaler', StandardScaler()),  # Scale features
   ('svm', SVC(class_weight='balanced', random_state=42, probability=True))  # SVM classifier with probability estimation
])

# Step 5: Define the parameter grid for GridSearchCV
param_grid = {
   'svm__kernel': ['rbf'],  # Focus on RBF kernel
   'svm__C': [1e-3, 0.1, 1, 10, 100],  # Regularization parameter
   'svm__gamma': [1e-3, 0.1, 1, 10]  # Kernel coefficient
}

# Step 6: Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_val, y_train_val)

# Step 7: Display the best parameters and best score
print("Best parameters from Grid Search:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Step 8: Evaluate the best model using K-Fold Cross-Validation on the training+validation set
best_model = grid_search.best_estimator_  # Extract the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # K-Fold Cross-Validation with 5 folds
scores = []
all_y_true = []
all_y_pred = []

print(f"\nPerforming {kf.get_n_splits()}-Fold Cross-Validation on the training+validation set:")

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
   precision = precision_score(y_val, y_pred_val, average='weighted')
   recall = recall_score(y_val, y_pred_val, average='weighted')
   f1 = f1_score(y_val, y_pred_val, average='weighted')

   scores.append(accuracy)

   # Print results for this fold
   print(f"\nFold {i + 1} Results (Validation Set):")
   print(f"Accuracy: {accuracy:.2f}")
   print(f"Precision: {precision:.2f}")
   print(f"Recall: {recall:.2f}")
   print(f"F1-Score: {f1:.2f}")
   print("Classification Report:")
   print(classification_report(y_val, y_pred_val))

# Step 9: Calculate and display overall performance
print("\nOverall Cross-Validation Results on Validation Set:")
print(f"Mean Accuracy: {np.mean(scores):.2f}")
print(f"Standard Deviation of Accuracy: {np.std(scores):.2f}")

# Step 10: Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Performance metrics on the test set
test_precision = precision_score(y_test, y_pred_test, average='weighted')
test_recall = recall_score(y_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')

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
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['ProductType'].unique(), yticklabels=df['ProductType'].unique())
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
