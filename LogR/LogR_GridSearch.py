# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

# Step 1: Load the data from the CSV file
data = pd.read_csv('Cleaned_BigMartWaste.csv')

# Step 2: Select the features and target variable
selected_features = ['Weight', 'FatContent', 'MRP']
X = data[selected_features]
y = data['ProductType']

# Step 3: Split the data into training+validation set (80%) and test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize Logistic Regression model with a pipeline that includes feature scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standard scaling for the features
    ('logreg', LogisticRegression(random_state=42, class_weight='balanced'))  # Logistic Regression model
])

# Step 5: Define the parameter grid for GridSearchCV
logreg_param_grid = {
    'logreg__C': np.logspace(-4, 4, 20),  # Regularization strength
    'logreg__solver': ['lbfgs', 'liblinear'],  # Solvers for optimization
    'logreg__max_iter': [1000, 2000, 3000],  # Maximum number of iterations
}

# Step 6: Perform GridSearchCV for hyperparameter tuning with K-Fold Cross-Validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=logreg_param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_val, y_train_val)

# Step 7: Display the best parameters and best score
print("Best parameters from Grid Search:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Step 8: Perform K-Fold Cross-Validation on the training+validation set
best_model = grid_search.best_estimator_  # Extract the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
all_y_true = []
all_y_pred = []

for i, (train_index, val_index) in enumerate(kf.split(X_train_val)):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    # Fit the model and make predictions on the validation set
    best_model.fit(X_train, y_train)
    y_pred_val = best_model.predict(X_val)

    # Store true and predicted labels
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred_val)

    # Calculate and print performance metrics for each fold
    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)

    scores.append(accuracy)
    print(f"\nFold {i + 1} Results (Validation Set):")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

# Step 9: Calculate and display overall performance across folds
print("\nOverall Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(scores):.2f}")
print(f"Standard Deviation of Accuracy: {np.std(scores):.2f}")

# Step 10: Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

print("\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.2f}, Test Precision: {test_precision:.2f}, Test Recall: {test_recall:.2f}, Test F1-Score: {test_f1:.2f}")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test, zero_division=0))

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
