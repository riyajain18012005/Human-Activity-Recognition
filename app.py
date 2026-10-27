import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
st.title("Human Activity Recognition Dashboard")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "har_artifacts")

LABELS_PATH = os.path.join(ARTIFACTS_DIR, "activity_labels.json")
COMPARISON_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.csv")

# ---------------- ACTIVITY LABELS ----------------
with open(LABELS_PATH, "r") as file:
    activity_labels = json.load(file)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏃 HAR Dashboard")
st.sidebar.write("Human Activity Recognition")
page = st.sidebar.radio(
    "Choose a section",
    ["Dashboard", "Activity Prediction", "About Project"]
)
st.write("Dashboard is loading successfully!")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("🏃 Human Activity Recognition")
    st.write(
        "A Deep Learning project that compares CNN, Simple RNN, "
        "and LSTM models for recognizing human activities from sensor data."
    )

    st.divider()

    # Load comparison results
    comparison = pd.read_csv(COMPARISON_PATH)

    col1, col2, col3 = st.columns(3)

    col1.metric("Models Trained", "3")
    col2.metric("Activities", "6")

    best_model = comparison.loc[
        comparison["Test Accuracy (%)"].idxmax(), "Model"
    ]
    col3.metric("Best Model", str(best_model))

    st.subheader("📊 Model Performance Comparison")

    st.dataframe(comparison, use_container_width=True)

    st.subheader("Test Accuracy")

    accuracy_chart = comparison.set_index("Model")[
        "Test Accuracy (%)"
    ]

    st.bar_chart(accuracy_chart)

    st.subheader("Weighted F1 Score")

    f1_chart = comparison.set_index("Model")[
        "Weighted F1 (%)"
    ]

    st.bar_chart(f1_chart)

    st.subheader("Training Time")

    time_chart = comparison.set_index("Model")[
        "Training Time (seconds)"
    ]

    st.bar_chart(time_chart)

# ---------------- ACTIVITY PREDICTION ----------------
elif page == "Activity Prediction":

    st.title("🎯 Predict Human Activity")

    st.write(
        "Select a trained model and upload a CSV file containing "
        "128 sensor readings and 9 sensor features."
    )

    st.info(
        "The CSV must contain exactly 128 rows and 9 columns, "
        "without a header row. Each column represents one sensor feature."
    )

    model_choice = st.selectbox(
        "Choose a model",
        ["CNN", "RNN", "LSTM"]
    )

    uploaded_file = st.file_uploader(
        "Upload sensor data CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            sensor_data = pd.read_csv(
                uploaded_file,
                header=None
            )

            if sensor_data.shape != (128, 9):
                st.error(
                    f"Invalid shape: {sensor_data.shape}. "
                    "Please upload a CSV with exactly 128 rows and 9 columns."
                )

            elif sensor_data.isnull().values.any():
                st.error("The CSV contains missing values.")

            else:
                sensor_array = sensor_data.to_numpy(dtype=np.float32)

                if not np.isfinite(sensor_array).all():
                    st.error("The CSV contains invalid numeric values.")

                else:
                    st.subheader("Sensor Data Preview")
                    st.dataframe(sensor_data.head(10), use_container_width=True)

                    if st.button("Predict Activity"):

                        model_files = {
                            "CNN": "cnn_model.keras",
                            "RNN": "rnn_model.keras",
                            "LSTM": "lstm_model.keras"
                        }

                        model_path = os.path.join(
                            ARTIFACTS_DIR,
                            model_files[model_choice]
                        )

                        with st.spinner("Analyzing sensor data..."):
                            model = load_model(model_path)

                            input_data = np.expand_dims(
                                sensor_array,
                                axis=0
                            )

                            probabilities = model.predict(
                                input_data,
                                verbose=0
                            )[0]

                            predicted_index = int(
                                np.argmax(probabilities)
                            )

                            predicted_activity = activity_labels[
                                str(predicted_index + 1)
                            ]

                            confidence = float(
                                probabilities[predicted_index] * 100
                            )

                        st.success(
                            f"Predicted Activity: {predicted_activity}"
                        )

                        st.metric(
                            "Prediction Confidence",
                            f"{confidence:.2f}%"
                        )

                        st.subheader("Class-wise Prediction Scores")

                        score_df = pd.DataFrame({
                            "Activity": [
                                activity_labels[str(i + 1)]
                                for i in range(len(probabilities))
                            ],
                            "Score (%)": probabilities * 100
                        })

                        st.bar_chart(
                            score_df.set_index("Activity")
                        )

        except Exception as error:
            st.error(f"Unable to process the file: {error}")

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.title("📘 About the Project")

    st.write("""
    **Human Activity Recognition (HAR)** uses sensor measurements
    to identify activities performed by a person.

    ### Activities
    - Walking
    - Walking Upstairs
    - Walking Downstairs
    - Sitting
    - Standing
    - Laying

    ### Deep Learning Models
    - **CNN:** Learns patterns from sensor sequences using convolution layers.
    - **Simple RNN:** Processes sequential sensor readings.
    - **LSTM:** Uses memory mechanisms to learn patterns across sequences.

    ### Dataset
    UCI Human Activity Recognition Using Smartphones Dataset.

    ### Input
    A sequence of 128 readings with 9 sensor features.

    ### Output
    The predicted activity and model confidence scores.
    """)

st.sidebar.divider()
st.sidebar.caption("Human Activity Recognition | Deep Learning Project")