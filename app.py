import numpy as np
import streamlit as st

from model import load_model


st.set_page_config(page_title="Health Assistant for Nepal")

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


symptoms = [
    "Fever",
    "Cough",
    "Fatigue",
    "Headache",
    "Breathing Issue",
    "Nausea",
    "Weight Loss",
    "Sore Throat",
    "Body Pain",
    "Vomiting",
    "Diarrhea",
    "Rash",
    "Chest Pain",
    "Dizziness",
    "Stomach Pain",
    "Runny Nose",
    "Sneezing",
    "Eye Pain",
    "Joint Pain",
    "Cut / Wound",
    "Dirty or Greasy Wound",
    "Redness / Swelling",
    "Pus",
    "Bleeding",
    "Burning Urination",
]


def get_message_hint(message):
    message = message.lower()

    if "fever" in message and "headache" in message:
        return "Fever and headache can happen in flu, dengue, typhoid, or viral fever. Select your symptoms below."
    if "cut" in message or "wound" in message:
        return "For a cut or wound, check if there is bleeding, dirt, swelling, or pus. Select the symptoms below."
    if "cough" in message or "breathing" in message:
        return "Cough or breathing problems can be linked to respiratory illness. Select your symptoms below."
    if "stomach" in message or "vomit" in message or "diarrhea" in message:
        return "Stomach pain, vomiting, or diarrhea can be linked to food poisoning or infection. Select your symptoms below."

    return "Now select the symptoms that match how you feel."


def make_input(selected_symptoms):
    return np.array([[
        int("Fever" in selected_symptoms),
        int("Cough" in selected_symptoms),
        int("Fatigue" in selected_symptoms),
        int("Headache" in selected_symptoms),
        int("Breathing Issue" in selected_symptoms),
        int("Nausea" in selected_symptoms),
        int("Weight Loss" in selected_symptoms),
        int("Sore Throat" in selected_symptoms),
        int("Body Pain" in selected_symptoms),
        int("Vomiting" in selected_symptoms),
        int("Diarrhea" in selected_symptoms),
        int("Rash" in selected_symptoms),
        int("Chest Pain" in selected_symptoms),
        int("Dizziness" in selected_symptoms),
        int("Stomach Pain" in selected_symptoms),
        int("Runny Nose" in selected_symptoms),
        int("Sneezing" in selected_symptoms),
        int("Eye Pain" in selected_symptoms),
        int("Joint Pain" in selected_symptoms),
        int("Cut / Wound" in selected_symptoms),
        int("Dirty or Greasy Wound" in selected_symptoms),
        int("Redness / Swelling" in selected_symptoms),
        int("Pus" in selected_symptoms),
        int("Bleeding" in selected_symptoms),
        int("Burning Urination" in selected_symptoms),
    ]])


st.title("Health Assistant for Nepal")
st.write("This app gives basic health guidance based on selected symptoms.")

model = load_model()

st.subheader("Describe how you feel")
message = st.text_input("Write your symptoms", placeholder="Example: I have fever and headache")

if message:
    st.info(get_message_hint(message))

st.subheader("Select symptoms")
selected_symptoms = st.multiselect("Select symptoms", symptoms, label_visibility="collapsed")

input_data = make_input(selected_symptoms)
symptom_count = int(input_data.sum())

if st.button("Analyze"):
    if symptom_count == 0:
        st.info("Please select at least one symptom.")
        st.stop()

    probabilities = model.predict_proba(input_data)[0]
    diseases = model.classes_
    top_indexes = np.argsort(probabilities)[-3:][::-1]

    st.subheader("Result")
    st.write("Most likely condition:", diseases[top_indexes[0]])
    st.write(f"Confidence: {probabilities[top_indexes[0]] * 100:.1f}%")

    st.write("Other possible conditions:")
    for index in top_indexes:
        percent = probabilities[index] * 100
        if percent > 0:
            st.write(f"- {diseases[index]}: {percent:.1f}%")

    st.subheader("Why this result")
    st.write("The result was influenced by these symptoms:")
    st.write(", ".join(selected_symptoms))

    st.subheader("Urgency")
    if "Breathing Issue" in selected_symptoms or "Chest Pain" in selected_symptoms:
        st.error("Urgent: Please seek medical help as soon as possible.")
    elif "Pus" in selected_symptoms or "Dirty or Greasy Wound" in selected_symptoms:
        st.error("A dirty wound or pus can be serious. Please visit a health post or hospital.")
    elif "Cut / Wound" in selected_symptoms and "Bleeding" in selected_symptoms:
        st.warning("Apply pressure with a clean cloth. If bleeding does not stop, get medical help.")
    elif symptom_count >= 5:
        st.warning("You selected many symptoms. It is better to consult a doctor.")
    elif symptom_count >= 3:
        st.warning("Monitor your symptoms and consider visiting a clinic.")
    else:
        st.info("Your symptoms may be mild, but keep monitoring them.")

    st.subheader("Basic advice")
    if symptom_count <= 2:
        st.write("Rest, drink enough water, and watch your symptoms.")
    elif symptom_count <= 4:
        st.write("Monitor your symptoms closely and consider consulting a doctor.")
    else:
        st.write("Please visit a healthcare professional.")

    st.subheader("Where to get help")
    st.write("For serious symptoms, visit the nearest health post, clinic, or hospital.")
    st.write("For emergency care, go to the nearest hospital or contact local health services.")

    st.subheader("Important note")
    st.write(
        "This app is for education only. It is not a doctor and should not replace "
        "professional medical advice, diagnosis, or treatment."
    )
