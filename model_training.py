import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the prepared dataset
df = pd.read_csv('prepared_breast_cancer_data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the ANN model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "breast_cancer_model.pkl")
print("✅ Model trained and saved successfully as 'breast_cancer_model.pkl'!")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2f}")
