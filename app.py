import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import utills
import joblib
from huggingface_hub import hf_hub_download
import os

with open('features.json', 'r') as file:
    col_map = json.load(file)

print("Initializing Streamlit UI...")
utills.streamlit_layout()
utills.css_markdown()

# Load trained model and preprocessing tools
print("Load trained model and preprocessing tools...")
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Dwaipayan08/random_forest_clinical_diabetes",
        filename="rf_model.pkl"
    )
    return joblib.load(model_path)

model = load_model()
# model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_details = joblib.load("model_details.pkl")

# ----------------- Streamlit UI -----------------------

tab1, tab2, tab3 = st.tabs(["📝 Input Form", "📊 Results", "Model Specifications"])
with tab1:
    st.subheader("Fill Patient Information")
    user_input = pd.DataFrame(columns=feature_names)

    col1, col2, col3 = st.columns(3)
    col_list = [col1, col2, col3]
    user_input = utills.get_user_input(feature_names, col_map, col_list)

    if st.button("🔍 Predict Diabetes Risk"):
        with col2:
                input_df = user_input
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]

                st.session_state["prediction"] = prediction
                st.session_state["prob"] = prob
                st.session_state["input"] = user_input.to_dict()
        st.markdown(f"User Input: ")
        st.table(user_input)

with tab2:
    if "prediction" in st.session_state:
        st.subheader("🔍 Prediction Result")
        st.markdown(f"### {'🟥 Diabetes Risk Detected' if st.session_state['prediction'] else '🟩 No Diabetes Risk'}")
        st.write(f"**Risk Level**: {st.session_state['prob']*100:.2f}%")

        # OpenAI Suggestion
        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
        )

        prompt = f"""
        Based on the below user data: Give some medical health advice related to diabetes and the measures this person should take
        to prevent diabtes in later life. \n
        Patient data: {st.session_state["input"]} \n
        Context: Based on a random forest classifier model, 
        the probability of this user getting diabetes in later life came out to be {st.session_state['prob']}. \n
        If diabetic risk is high, provide 3 practical suggestions. Keep the tone supportive and non-clinical.
        """

        try:
            response = client.responses.create(
            model="gpt-4o",
            instructions="You are a medical expert and is consulting an user.",
            input=prompt,
            temperature=0.7,
            max_output_tokens=400
            )
            
            suggestion = response.output_text
            st.subheader("💡 AI Health Suggestions")
            st.write(suggestion)
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")

        st.warning("Note: Results are not a substitute for medical advice.")
    else:
        st.info("Run a prediction from the 'Input Form' tab.")

with tab3:
    st.subheader("Below are the specifications of the backend model")
    st.markdown(f"""
                Model Used: LightGBM \n
                Accuracy of the model: 71.6% \n
                Recall score of the model: 76.5% \n
                F1 Score: 45.2% \n """)
