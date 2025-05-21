try:
    import sklearn
    print("scikit-learn is available!")
except ImportError:
    print("scikit-learn is not installed!")

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

    
# Load dataset and train model
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Streamlit UI
st.title("Breast Cancer Prediction App")
st.write("Input the 30 features to predict if the tumor is **malignant (0)** or **benign (1)**.")

# Input features
input_values = []
for feature in data.feature_names:
    val = st.number_input(f"{feature}", value=float(np.mean(X[feature])))
    input_values.append(val)

# Prediction
if st.button("Predict"):
    input_scaled = scaler.transform([input_values])
    prediction = model.predict(input_scaled)[0]
    result = "Benign (1)" if prediction == 1 else "Malignant (0)"
    st.success(f" Prediction: **{result}**")
