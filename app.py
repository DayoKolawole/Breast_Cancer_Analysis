import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("🩺 Breast Cancer Prediction App")

# ✅ Load the trained model
model = None
try:
    model = joblib.load("breast_cancer_model.pkl")
    st.write("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.error("⚠️ Model file not found! Ensure 'breast_cancer_model.pkl' is in the project folder.")

# ✅ Load dataset
df = None
try:
    df = pd.read_csv('prepared_breast_cancer_data.csv')
    st.write("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("⚠️ Dataset file not found! Ensure 'prepared_breast_cancer_data.csv' is in the project folder.")

# 🛑 Ensure that both the dataset and model are loaded before proceeding
if df is not None and model is not None:

    # 📊 Feature Distribution Visualization
    st.subheader("📊 Feature Distribution")

    # Select a feature to visualize
    feature = st.selectbox("Select a Feature for Distribution", df.drop(columns=['target']).columns)

    # Plot the histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Sidebar input fields
    st.sidebar.header("🔍 Enter Tumor Features")

    # Get feature names (all columns except 'target')
    feature_names = df.drop(columns=['target']).columns

    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.sidebar.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

    # 🔮 Predict button
    if st.sidebar.button("🔮 Predict"):
        # Convert dictionary to DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # 🛑 Ensure model is loaded before making predictions
        if model is not None:
            prediction = model.predict(input_df)[0]

            # Display result
            result = "🟢 Benign (No Cancer)" if prediction == 1 else "🔴 Malignant (Cancer Present)"
            st.subheader("🔍 Prediction Result")
            st.success(f"**Prediction:** {result}")
        else:
            st.error("⚠️ Model not available for prediction!")

    # 📊 Show Model Accuracy
    st.write("---")
    st.subheader("📊 Model Performance")

    # Load true values for accuracy
    y = df['target']
    X = df.drop(columns=['target'])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🛑 Ensure model is loaded before evaluating accuracy
    if model is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Model Accuracy:** {accuracy:.2f}")
        st.write("This accuracy is based on the test dataset using the trained ANN model.")
    else:
        st.error("⚠️ Model not available for accuracy calculation.")







