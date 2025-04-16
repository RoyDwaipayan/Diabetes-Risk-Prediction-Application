import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import urllib.request
import os

def get_image_base64(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

def download_model(url, filename="rf_model.pkl"):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

def streamlit_layout():
    st.set_page_config(
    page_title="Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="wide"
    )

    image_base64 = get_image_base64("Cover_pic.png")  
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{image_base64}" style="width: 1600px; max-width: 100%; height: auto;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

def css_markdown():
    st.markdown("""
    <style>
    .main-title {
        font-size: 36px !important;
        color: #0077b6;
        text-align: center;
    }
    .stButton > button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
    }
                
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:18px;
    font-weight: bold;
    color: #004085;
    }
    
    </style>
    <br>
    """, unsafe_allow_html=True)

def get_user_input(feature_names, col_map, col_list):

    model_details = joblib.load("model_details.pkl")

    binary_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
    numeric_cols = ['GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income','BMI_box']

    user_input = pd.DataFrame(columns=feature_names)

    for i, feature in enumerate(feature_names):
        label = col_map[feature]
        if feature in binary_cols:  # Binary
            with col_list[i % 3]:
                if feature == 'Sex':
                    val = st.radio(label, ["Male", "Female"], horizontal=True)
                    user_input.loc[0, feature] = 1 if val == "Male" else 0
                else: 
                    val = st.radio(label, ["No", "Yes"], horizontal=True)
                    user_input.loc[0, feature] = 1 if val == "Yes" else 0
        elif feature in numeric_cols:  # Numeric
            step = 1
            with col_list[i % 3]:
                if feature == 'BMI_box':
                    val = st.slider(label, min_value=0, max_value=100, step=step,
                                    help="BMI = weight (lb) / height² (inches) * 703")
                    user_input.loc[0, feature] = apply_boxcox(val, model_details["Fitted Lambda"]["BMI"])
                elif feature == 'GenHlth':
                    val = st.slider(label, min_value=0, max_value=5, step=step,
                                    help="Would you say that in general your health is: scale " \
                                        "1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor")
                    user_input.loc[0, feature] = val
                elif feature == 'MentHlth':
                    val = st.slider(label, min_value=0, max_value=30, step=step,
                                    help="Thinking about your mental health, which includes stress, depression, " \
                                    "and problems with emotions, for how many days during the past 30 days was " \
                                    "your mental health not good? scale 1-30 days")
                    user_input.loc[0, feature] = val
                elif feature == 'PhysHlth':
                    val = st.slider(label, min_value=0, max_value=30, step=step,
                                    help="thinking about your physical health, which includes physical illness "
                                    "and injury, for how many days during the past 30 days was your physical " \
                                    "health not good? scale 1-30 days")
                    user_input.loc[0, feature] = val
                elif feature == 'Age':
                    val = st.slider(label, min_value=0, max_value=13, step=step,
                                    help="13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older")
                    user_input.loc[0, feature] = val
                elif feature == 'Education':
                    val = st.slider(label, min_value=0, max_value=6, step=step,
                                    help="Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only " \
                                    "kindergarten , 2 = Grades 1 through 8 (Elementary) ,3 = Grades 9 throug 11 (Some high school), " \
                                    "4 = Grade 12 or GED (High school graduate) , 5 = College 1 year to 3 years (Some college or technical school) " \
                                    ", 6 = College 4 years or more (College graduate).")
                    user_input.loc[0, feature] = val
                elif feature == 'Income':
                    val = st.slider(label, min_value=0, max_value=13, step=step,
                                    help="1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more")
                    user_input.loc[0, feature] = val
                else:
                    val = st.slider(label, min_value=0, max_value=100, step=step)
                    user_input.loc[0, feature] = val

    return user_input


def apply_boxcox(x, lmbda):
    if lmbda == 0:
        return np.log(x + 0.001)
    else:
        return ((x + 0.001) ** lmbda - 1) / lmbda