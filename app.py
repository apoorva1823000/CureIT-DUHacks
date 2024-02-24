# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:06:49 2024

@author: Apoorva .S. Mehta
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# loading the models
diabetes_model = pickle.load(open('diabetes_trained_model.sav','rb'))
heart_disease_model = pickle.load(open('heart_disease_trained_model.sav','rb'))
parkinsons_disease_model = pickle.load(open('Parkinsons_trained_model.sav','rb'))
bc_disease_model = pickle.load(open('breast_cancer_model.sav','rb'))

st.set_page_config(
        page_title="CureIT - The Healthcare Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "mailto:apoorva.cse1@gmail.com",
            'About': "Created by team LoopBreakers, CureIt is an advance healthcare support featuring machine learning to diagnose your statistical health reports, analyse them and predict the possibility of various diseases"
        }
    )

def breast_cancer():
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab.")
        data = {
           'Feature': [
               'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
               'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
               'Radius Error', 'Texture Error', 'Perimeter Error', 'Area Error', 'Smoothness Error',
               'Compactness Error', 'Concavity Error', 'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
               'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
               'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
           ],
           'Description': [
               'Mean of distances from center to points on the perimeter',
               'Standard deviation of gray-scale values',
               'Mean of perimeter of cell nuclei',
               'Mean of area of cell nuclei',
               'Mean of local variation in radius lengths',
               'Mean of local variation in gray-scale values',
               'Mean of local variation in area of cell nuclei',
               'Mean of local variation in radius lengths of concave portions of contour',
               'Mean of local variation in radius lengths',
               'Mean of local variation in area of cell nuclei',
               'Standard error of the mean of distances from center to points on the perimeter',
               'Standard error of the mean of gray-scale values',
               'Standard error of the mean of perimeter of cell nuclei',
               'Standard error of the mean of area of cell nuclei',
               'Standard error of the mean of local variation in radius lengths',
               'Standard error of the mean of local variation in gray-scale values',
               'Standard error of the mean of local variation in area of cell nuclei',
               'Standard error of the mean of local variation in radius lengths of concave portions of contour',
               'Standard error of the mean of local variation in radius lengths',
               'Standard error of the mean of local variation in area of cell nuclei',
               'Largest mean value of the distance from center to points on the perimeter',
               'Largest mean value of gray-scale values',
               'Largest mean value of perimeter of cell nuclei',
               'Largest mean value of area of cell nuclei',
               'Largest mean value of local variation in radius lengths',
               'Largest mean value of local variation in gray-scale values',
               'Largest mean value of local variation in area of cell nuclei',
               'Largest mean value of local variation in radius lengths of concave portions of contour',
               'Largest mean value of local variation in radius lengths',
               'Largest mean value of local variation in area of cell nuclei'
           ]
       }
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )
        
def diabetes_predictor():
    with st.container():
        st.title("Diabetes Predictor")
        st.write("Please connect this app to your healthcare provider to predict your risk of developing diabetes based on various health factors. This app utilizes a machine learning model to assess the likelihood of diabetes onset.")
        data = {
        'Attribute': [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
            ],
        'Description': [
            'Number of times pregnant',
            'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
            'Diastolic blood pressure (mm Hg)',
            'Triceps skin fold thickness (mm)',
            '2-Hour serum insulin (mu U/ml)',
            'Body mass index (weight in kg/(height in m)^2)',
            'Diabetes pedigree function',
            'Age (years)',
            'Class variable (0 or 1) indicating whether the individual has diabetes or not'
            ]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
            }])
        )   

def heart_disease_predictor():
    with st.container():
        st.title("Heart Disease Predictor")
        st.write("Connect this app to your cardiovascular specialist to assess your risk of heart disease based on medical data. Using a machine learning model, this app predicts the probability of heart disease occurrence.")
        data = {
        'Attribute': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ],
        'Description': [
            'Age (in years)',
            'Sex (0 = female, 1 = male)',
            'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
            'Resting blood pressure (in mm Hg)',
            'Serum cholesterol (in mg/dl)',
            'Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)',
            'Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)',
            'Maximum heart rate achieved',
            'Exercise induced angina (0 = no, 1 = yes)',
            'ST depression induced by exercise relative to rest',
            'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
            'Number of major vessels (0-3) colored by flourosopy',
            'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)',
            'Presence of heart disease (0 = no, 1 = yes)'
        ]
    }

        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )

def parkinsons_predictor_model():
    with st.container():
        st.title("Parkinsons Detector")
        st.write("Link this app to your neurologist to predict the likelihood of Parkinsons disease based on clinical data. With a machine learning algorithm, this app estimates the probability of Parkinsons onset.")
        data = {
        'Feature': [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR',
            'RPDE', 'D2',
            'DFA',
            'spread1', 'spread2', 'PPE'
        ],
        'Description': [
            'Average vocal fundamental frequency',
            'Maximum vocal fundamental frequency',
            'Minimum vocal fundamental frequency',
            'Several measures of variation in fundamental frequency',
            'Several measures of variation in fundamental frequency (absolute)',
            'Variation in fundamental frequency - Relative amplitude perturbation',
            'Variation in fundamental frequency - Period perturbation quotient',
            'Variation in fundamental frequency - Jitter:DDP',
            'Several measures of variation in amplitude',
            'Several measures of variation in amplitude (in dB)',
            'Amplitude perturbation quotient - Three-point average',
            'Amplitude perturbation quotient - Five-point average',
            'Variation in amplitude',
            'Amplitude perturbation quotient - Three-point average (in dB)',
            'Noise to tonal components ratio',
            'Harmonic to noise ratio',
            'Nonlinear dynamical complexity measure',
            'Nonlinear dynamical complexity measure',
            'Signal fractal scaling exponent',
            'Nonlinear measure of fundamental frequency variation',
            'Nonlinear measure of fundamental frequency variation',
            'Nonlinear measure of fundamental frequency variation'
        ]
    }

    # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )
        
        

#sidebar menu
with st.sidebar:
    st.sidebar.header("Text based predictive system")
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes',
                            'Heart Disease',
                            'Parkinsons',
                            'Breast Cancer',], 
                           icons = ['clipboard-plus','heart-pulse','person-badge-fill','activity'],
                           default_index=0)
    
    
#Setting theme

st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        .sidebar .sidebar-content .block-container {
            padding: 1rem;
        }
        .sidebar .sidebar-content .stButton > button {
            background-color: #1abc9c;
            color: white;
            font-weight: bold;
        }
        .sidebar .sidebar-content .stButton > button:hover {
            background-color: #16a085;
        }
        .main .block-container {
            padding: 2rem;
        }
        .main .block-container .stTextInput {
            margin-bottom: 1rem;
        }
        .main .block-container .stButton > button {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        .main .block-container .stButton > button:hover {
            background-color: #2980b9;
        }
        .main .st-success {
            color: green;
        }
        .main .stTable {
            max-width: 100%;
            font-size: 1vw;
        }
        .main .stTable tr:hover {
            background-color: #ffff99;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Working with the models

# Diabetes Model
if (selected=='Diabetes'):
    diabetes_predictor()
    st.title('Diabetes Prediction using ML')
    
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age Value')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
        BMI = st.text_input('BMI Level')
        
    
    fields_filled = all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    if not fields_filled:
        st.error("Please fill in all the fields.")
    else:
        diabetes_diagnosis = ''
        #creating a button for prediction
        if st.button('Diabetes Test Results'):
            diabetes_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(diabetes_prediction[0]==1):
            diabetes_diagnosis = 'The person has high chances of being diagnosed as a Diabetic'
        else:
            diabetes_diagnosis = 'The person is most likely to be Non-Diabetic'
        st.success(diabetes_diagnosis)
    
# Heart Disease Model
if (selected=='Heart Disease'):
    heart_disease_predictor()
    st.title('Heart Disease Prediction using ML')
    
    
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.text_input('Enter your Age')
        trest = st.text_input('Resting Blood Pressure Value')
        restecg = st.text_input('Resting electrocardiographic results ')
        oldpeak = st.text_input('OldPeak Value')
        thal = st.text_input('Thal Value')
    with col2:
        sex = st.text_input('Enter 0 if Female and 1 if Male')
        chol = st.text_input('Cholestrol Level')
        thalach = st.text_input('Maximum heart rate achieved Value')
        slope = st.text_input('Slope Value')
    with col3:
        cp = st.text_input('CP Value')
        fbs = st.text_input('Fasting Blood Sugar Value')
        exang = st.text_input('Exercise Induced Angina')
        ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
        
        
    fields_filled = all([age,sex,cp,trest,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        heart_disease_diagnosis = ''
        #creating a button for prediction
        if st.button('Heart Disease Test Results'):
            heart_disease_prediction = heart_disease_model.predict([[age,sex,cp,trest,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if(heart_disease_prediction[0]==1):
            heart_disease_diagnosis = 'The person has chances of getting a Heart Disease'
        else:
            heart_disease_diagnosis = 'The person has low chances of getting a Heart Disease'
        st.success(heart_disease_diagnosis)
    
# Parkinson's Prediction
if (selected=='Parkinsons'):
    parkinsons_predictor_model()
    st.title('Parkinsons Prediction using ML')

    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Fo = st.text_input('Enter MDVP:Fo value')
        Jitter = st.text_input('Enter MDVP:Jitter value')
        PPQ = st.text_input('Enter MDVP:PPQ value')
        Shimmer2 = st.text_input('Enter the Shimmer(db) Value')
        APQ = st.text_input('Enter the MDVP(APQ) Value')
        HNR = st.text_input('Enter the HNR Value')
        spread1 = st.text_input('Enter the Spread1 Value')
        PPE = st.text_input('Enter the PPE Value')
    with col2:
        Fhi = st.text_input('Enter MDVP:Fhi value')
        Jitter2 = st.text_input('Enter Jitter(Abs) value')
        Jitter3 = st.text_input('Enter Jitter(DDP) value')
        APQ3 = st.text_input('Enter the Shimmer(APQ3) Value')
        DDA = st.text_input('Enter the Shimmer (DDA) Value')
        RPDE = st.text_input('Enter the RPDE Value')
        spread2 = st.text_input('Enter the Spread2 Value')
    with col3:
        Flo = st.text_input('Enter MDVP:Flo value')
        RAP = st.text_input('Enter MDVP:RAP value')
        Shimmer = st.text_input('Enter MDVP(Shimmer) Value')
        APQ5 = st.text_input('Enter the Shimmer(APQ5) Value')
        NHR = st.text_input('Enter the NHR Value')
        DFA = st.text_input('Enter the DFA Value')
        D2 = st.text_input('Enter the D2 Value')
        
    fields_filled = all([Fo,Fhi,Flo,Jitter,Jitter2,RAP,PPQ,Jitter3,Shimmer,Shimmer2,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        parkinsons_disease_diagnosis = ''
        #creating a button for prediction
        if st.button('Parkinson Disease Test Results'):
            parkinsons_disease_prediction = parkinsons_disease_model.predict([[Fo,Fhi,Flo,Jitter,Jitter2,RAP,PPQ,Jitter3,Shimmer,Shimmer2,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        if(parkinsons_disease_prediction[0]==1):
            parkinsons_disease_diagnosis = 'The person has high chances of Parkinsons Disease'
        else:
            parkinsons_disease_diagnosis = 'The person has low chances of Parkinsons Disease'
        st.success(parkinsons_disease_diagnosis)
    
# Breast Cancer Model
if (selected=='Breast Cancer'):
    breast_cancer()
    st.title('Breast Cancer Prediction using ML')
    col1,col2,col3 = st.columns(3)
    
    with col1:
        mean_radius = st.text_input('Mean Radius')
        mean_texture = st.text_input('Mean Texture')
        mean_perimeter = st.text_input('Mean Perimeter')
        mean_area = st.text_input('Mean Area')
        mean_smoothness = st.text_input('Mean Smoothness')
        mean_compactness = st.text_input('Mean Compactness')
        mean_concavity = st.text_input('Mean Concavity')
        mean_concave_points = st.text_input('Mean Concave Points')
        mean_symmetry = st.text_input('Mean Symmetry')
        mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
    with col2:
        radius_error = st.text_input('Radius Error')
        texture_error = st.text_input('Texture Error')
        perimeter_error = st.text_input('Perimeter Error')
        area_error = st.text_input('Area Error')
        smoothness_error = st.text_input('Smoothness Error')
        compactness_error = st.text_input('Compactness Error')
        concavity_error = st.text_input('Concavity Error')
        concave_points_error = st.text_input('Concave Points Error')
        symmetry_error = st.text_input('Symmetry Error')
        fractal_dimension_error = st.text_input('Fractal Dimension Error')
    with col3:
        worst_radius = st.text_input('Worst Radius')
        worst_texture = st.text_input('Worst Texture')
        worst_perimeter = st.text_input('Worst Perimeter')
        worst_area = st.text_input('Worst Area')
        worst_smoothness = st.text_input('Worst Smoothness')
        worst_compactness = st.text_input('Worst Compactness')
        worst_concavity = st.text_input('Worst Concavity')
        worst_concave_points = st.text_input('Worst Concave Points')
        worst_symmetry = st.text_input('Worst Symmetry')
        worst_fractal_dimension = st.text_input('Worst Fractal Dimension')
        
    fields_filled = all([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error, smoothness_error,
        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        breast_cancer_disease_diagnosis = ''
        #creating a button for prediction
        if st.button('Breast Cancer Disease Test Results'):
            bc_disease_prediction = bc_disease_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                                                               mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
                                                               radius_error, texture_error, perimeter_error, area_error, smoothness_error,
                                                               compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                                                               worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
                                                               worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]])
        if(bc_disease_prediction[0]==1):
            bc_disease_diagnosis = 'The tumor is observed to have traits of being Benign'
        else:
            parkinsons_disease_diagnosis = 'The tumor is observed to have traits of being Malignant'
        st.success(breast_cancer_disease_diagnosis)