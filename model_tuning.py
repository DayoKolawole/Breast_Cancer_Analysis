import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

# Load the prepared dataset
df = pd.read_csv('prepared_breast_cancer_data.csv')

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up parameter grid for Grid Search CV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [200, 500],
    'learning_rate': ['constant', 'adaptive']
}

# Initialize model and Grid Search
model = MLPClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("\nBest Parameters Found:\n", grid_search.best_params_)
print("\nBest Cross-Validation Score:", grid_search.best_score_)

# Evaluate best model on test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"\nTest Set Accuracy with Tuned Model: {test_accuracy:.2f}")
