import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Cancer Predictor")

st.title("Breast Cancer Prediction")

df = pd.read_csv("data.csv")

df = df.drop(columns=["id"], errors="ignore")

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

df = df[['diagnosis'] + features]

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Train Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model trained")

if st.session_state.ready:

    st.subheader("Enter Values")

    input_data = []

    for col in X.columns:
        val = st.number_input(col, value=float(X[col].mean()))
        input_data.append(val)

    if st.button("Predict"):

        input_array = np.array([input_data])
        prediction = st.session_state.model.predict(input_array)[0]

        if prediction == 1:
            st.error("Malignant (Cancer Detected)")
        else:
            st.success("Benign (No Cancer)")

else:
    st.info("Train the model first")