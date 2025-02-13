# data_preparation.py
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target variable
df['target'] = data.target

# Display the first few rows
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values per Column:\n", missing_values)

from sklearn.feature_selection import SelectKBest, f_classif

# Feature selection: Select the top 10 features
X = df.drop('target', axis=1)
y = df['target']

# Apply SelectKBest with ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("\nTop 10 Selected Features:\n", selected_features)

# Prepare dataset with selected features for modeling
X_selected_df = df[selected_features]
y = df['target']

# Display the prepared dataset
print("\nPrepared Dataset:\n", X_selected_df.head())
print("\nTarget Variable:\n", y.head())

# Save prepared dataset for modeling
X_selected_df['target'] = y
X_selected_df.to_csv('prepared_breast_cancer_data.csv', index=False)
print("\nDataset saved as 'prepared_breast_cancer_data.csv'")

