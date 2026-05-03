import streamlit as st

from health_assistant.config import SYMPTOMS
from health_assistant.model import load_model
from health_assistant.predict import (
    create_symptom_vector,
    format_prediction_results,
    get_advice_message,
    get_urgency_message,
    get_user_hint,
)


def render_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #111827;
        }

        .block-container {
            max-width: 760px;
            padding-top: 48px;
            padding-bottom: 48px;
        }

        h1, h2, h3 {
            color: #f9fafb;
        }

        p, label, span, div {
            color: #e5e7eb;
        }

        [data-testid="stTextInput"] input {
            background-color: #1f2937;
            color: #f9fafb;
            border: 1px solid #374151;
            border-radius: 6px;
        }

        [data-testid="stMultiSelect"] {
            background-color: #1f2937;
            border-radius: 6px;
        }

        [data-testid="stAlert"] {
            border-radius: 6px;
        }

        .stButton button {
            background-color: #14b8a6;
            color: #06201d;
            border: none;
            border-radius: 6px;
            font-weight: 600;
        }

        .stButton button:hover {
            background-color: #2dd4bf;
            color: #06201d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Health Assistant for Nepal")
    render_style()

    st.title("Health Assistant for Nepal")
    st.write("This app gives basic health guidance based on selected symptoms.")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Required data or model files are missing. Run `python train_model.py` first.")
        st.stop()

    st.subheader("Describe how you feel")
    message = st.text_input("Write your symptoms", placeholder="Example: I have fever and headache")

    if message:
        st.info(get_user_hint(message))

    st.subheader("Select symptoms")
    selected_symptoms = st.multiselect("Select symptoms", SYMPTOMS, label_visibility="collapsed")
    input_data = create_symptom_vector(selected_symptoms)
    symptom_count = int(input_data.sum())

    if st.button("Analyze"):
        if symptom_count == 0:
            st.info("Please select at least one symptom.")
            return

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        results = format_prediction_results(probabilities, model.classes_)

        st.subheader("Result")
        st.write("Most likely condition:", results[0]["disease"])
        st.write(f"Confidence: {results[0]['confidence']:.1f}%")

        st.write("Other possible conditions:")
        for row in results:
            st.write(f"- {row['disease']}: {row['confidence']:.1f}%")

        st.subheader("Why this result")
        st.write("The result was influenced by these symptoms:")
        st.write(", ".join(selected_symptoms) or "No symptoms selected.")

        urgency_text, urgency_type = get_urgency_message(selected_symptoms, symptom_count)
        if urgency_type == "error":
            st.error(urgency_text)
        elif urgency_type == "warning":
            st.warning(urgency_text)
        else:
            st.info(urgency_text)

        st.subheader("Basic advice")
        st.write(get_advice_message(symptom_count))

        st.subheader("Where to get help")
        st.write("For serious symptoms, visit the nearest health post, clinic, or hospital.")
        st.write("For emergency care, go to the nearest hospital or contact local health services.")

        st.subheader("Important note")
        st.write(
            "This app is for education only. It is not a doctor and should not replace "
            "professional medical advice, diagnosis, or treatment."
        )


if __name__ == "__main__":
    main()
