import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from scipy import stats
from scipy.stats import boxcox
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu
from scipy.special import inv_boxcox
import warnings

# Set page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Insurance Tailorings",
    page_icon=r'insurance.jpeg',
)


# Injecting CSS for custom styling
st.markdown("""
    <style>
    /* Tabs */
    div.stTabs [data-baseweb="tab-list"] button {
        font-size: 25px;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px 20px;
        margin: 10px 2px;
        border-radius: 10px;
    }
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #009688;
        color: white;
    }

    div.stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3e8e41;
        color: white;
    }
    /* Button */
    .stButton>button {
        font-size: 22px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None:
        return None  # Handle missing value
    transformed_value = boxcox([value], lmbda=lmbda)[0]
    return transformed_value

def reverse_boxcox_transform(predicted, lambda_val):
    return inv_boxcox(predicted, lambda_val)

# Load the saved lambda values
with open(r'pkls/transformation/transformation_params.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

with open(r'pkls/edu_en.pkl', 'rb') as f:
    edu_pickle = pickle.load(f)

    
# with open(r'pkls/scale_reg.pkl', 'rb') as f:
    # scale_reg = pickle.load(f)
with open(r'pkls/K_means.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open(r'pkls/ADA_Classifier.pkl', 'rb') as f:
    Class_model = pickle.load(f)
    

with open(r'pkls/GB_Regressor.pkl', 'rb') as f:
    Reg_model = pickle.load(f)
    
    
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'About'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'should_navigate' not in st.session_state:
    st.session_state.should_navigate = False
 

    
with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Customer Profile Input', 'Customer Insights'],
        icons=['house-door-fill', 'bar-chart-fill'],
        menu_icon="cast",
        default_index=0,
        key='menu_option',
        styles={
            "container": {"padding": "12!important", "background-color": "gray"},
            "icon": {"color": "#000000", "font-size": "25px", "font-family": "Times New Roman"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "#ffffff", "color": "#55ACEE", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)

if st.session_state.should_navigate:
    selected = 'Customer Insights'
    st.session_state.should_navigate = False
    
st.markdown("<h1 style='text-align: center; font-size: 38px; color: #808080; font-weight: 700; font-family: inherit;'>Insurance Analytics & Insights</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-size: 26px; color: #808080; font-weight: 200; font-family: inherit;'>Data-Driven Solutions for Customer Segmentation and Fraud Detection</h4>", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "About":
    st.markdown("<h3 style='text-align: center; font-size: 38px; color: #ffffff; font-weight: 700; font-family: inherit;'>Understanding Insurance Insights</h3>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Overview </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;text-indent: 4em;'>
         The objective of this app is to leverage trained machine learning alogrithm to extract valuable insights from insurance data.
        By utilizing predictive models, this app provides data-driven insights to help insurance companies make Enhancing Decision-Making,
        Optimizing Risk Assessment, Improving Operational Efficiency.

</p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 34px; text-align: left; font-family: inherit; color: #ffffff;'> How to Use: </h3>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Customer Profile Input: </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;text-indent: 4em;'>
        Enter customer details such as age, gender, and policy information. The app will generate personalized segmentation and predictions.
</p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Customer Insights: </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;text-indent: 4em;'>
        See detailed customer segmentation, marketing strategies, risk profiles, and predictions for fraud detection and premium pricing based on the input data.
    </p>""", unsafe_allow_html=True)

    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;text-indent: 4em;'>
        Whether you're an underwriter, risk manager, or marketer, this app provides actionable insights to help you make data-driven decisions that benefit both your business and your customers.
    </p>""", unsafe_allow_html=True)

    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Contributing </h3>", unsafe_allow_html=True)
    github_url = "https://github.com/Santhosh-Analytics/Singapore-Resale-Flat-Prices-Predicting"
    st.markdown("""<p style='text-align: left; font-size: 18px;text-indent: 4em; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
    </p>""".format(github_url), unsafe_allow_html=True)

elif selected == "Customer Profile Input":
    st.title("Customer Profile")


    # Options for various dropdowns
    states = ('OH', 'IL', 'IN')
    policy_deduc_opt = (500, 1000, 1500, 2000)
    edu_opt = ('JD', 'High School', 'Associate', 'MD', 'Masters', 'PhD', 'College')
    occu_opt = ('machine-op-inspct', 'prof-specialty', 'tech-support', 'sales', 'exec-managerial', 'craft-repair',
             'transport-moving', 'other-service', 'priv-house-serv', 'armed-forces', 'adm-clerical', 'protective-serv',
              'handlers-cleaners', 'farming-fishing')
    hobbies_opt = ['reading', 'exercise', 'paintball', 'bungie-jumping', 'golf', 'movies', 'camping', 'kayaking',
         'yachting', 'hiking', 'video-games', 'base-jumping', 'skydiving', 'board-games', 'polo', 'chess', 'dancing',
          'sleeping', 'cross-fit', 'basketball']
    insured_opt = ['own-child', 'other-relative', 'not-in-family', 'husband', 'wife', 'unmarried']
    incident_opt = ['Multi-vehicle Collision', 'Single Vehicle Collision', 'Vehicle Theft', 'Parked Car']
    make_opt = ['Dodge', 'Suburu', 'Saab', 'Nissan', 'Chevrolet', 'BMW', 'Ford', 'Toyota', 'Audi', 'Accura',
         'Volkswagen', 'Jeep', 'Mercedes', 'Honda']
    collision_opt = ['Front Collision','Others','Rear Collision','Side Collision']

    col1, col, col2 = st.columns([2,.5,2])

    with col1:
        cust_month = st.number_input('Enter the Customer tenure ranges:', help="Enter the Customer tenure ranges. If new customer enter 0:",step = 1)
        policy_state = st.selectbox('Select policy State:', states    ,  help="Select Policy State/Location" )
        policy_deduc = st.selectbox('Select deductable:', policy_deduc_opt,  help="Portion of a claim that policy holder responsible to pay." )
        policy_premium = st.number_input('Enter annual premium amount:', help="Enter annual premium amount",step = 100)
        claim_amount = st.number_input('Enter claim amount:', help="Enter claim amount",step = 100)
        cust_age = st.number_input('Enter the Customer age:', help="Enter the Customer age:",step=1)
        insured_sex = st.selectbox('Select gender:', ['Male', 'Female'],  help="Customer Gender")
        education = st.selectbox('Select education:', edu_opt, help="Customer education Level")
        occupation = st.selectbox('Select occupation:', occu_opt,  help="Customer occupation")
        hobbies = st.selectbox('Select hobbies:', hobbies_opt,  help="Select hobbies") 
        insured = st.selectbox('Select insured relation:', insured_opt,  help="Select insured relation")
        fraud = st.selectbox('Select Fraud :', [True,False],  help="Select insured relation")

    with col2:
        auto_make = st.selectbox('Select auto make:', make_opt, help="Select Vehicle make")
        year = st.selectbox('Select make year:', [i for i in range(1994, 2016)],  help="Select Vehicle make year")
        incident_type = st.selectbox('Select incident type:', incident_opt,  help="Select incident type") 
        collision_type = st.selectbox('Select Collision type:', collision_opt, help="Select Collision type")

       
        st.write(' ')
        st.write(' ')
        button = st.button('Get Insights!')
        st.write(st.session_state.navigate_to_insights)
    
    
#     

    sex = {'Male': 1, 'Female':0}
    
    if education:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoded_edu = edu_pickle.transform([education])[0]
        except Exception as e:
            st.error(f"Error encoding education: {e}")
            encoded_edu = None  # or some default value
    
    coll_array = [0] * len(collision_opt)
    selected_index = collision_opt.index(collision_type)
    coll_array[selected_index]= 1
    coll_str = ', '.join(map(str, coll_array))
    # st.write(coll_str)
    
    inc_array = [0] * len(incident_opt)
    selected_inc_index = incident_opt.index(incident_type)
    inc_array[selected_inc_index]= 1
    inc_str = ', '.join(map(str, inc_array))
    # st.write(inc_str)
    
    rela_array = [0] * len(insured_opt)
    selected_rela_index = insured_opt.index(insured)
    rela_array[selected_rela_index]= 1
    rela_str = ', '.join(map(str, rela_array))
    # st.write(rela_str)
    
    hobbies_array = [0] * len(hobbies_opt)
    selected_hoobies_index = hobbies_opt.index(hobbies)
    hobbies_array[selected_hoobies_index]= 1
    hobbies_str = ', '.join(map(str, hobbies_array))
    # st.write(hobbies_str)
    
    st.write(encoded_edu)
    
    data = np.array([[cust_month,policy_deduc,sex[insured_sex],year,encoded_edu] + [cust_age] + coll_array + hobbies_array + rela_array + inc_array ])
    st.write(data)
    # scaled_data = scale_reg.transform(data)
    # st.write(scaled_data)
    
    st.session_state.data = data

    if button:
        if st.session_state.data is not None:
            try:
                # Your prediction logic here
                kmeans_prediction = kmeans.predict(st.session_state.data)
                # class_prediction = Class_model.predict(st.session_state.data)
                # reg_prediction = Reg_model.predict(st.session_state.data)
                
                # Store predictions in session state
                st.session_state.prediction = {
                    'kmeans': kmeans_prediction,
                    # 'classification': class_prediction,
                    # 'regression': reg_prediction
                }
                
                # Set flag to navigate to Customer Insights
                st.session_state.should_navigate = True
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter customer data before running the prediction.")
                
        # st.write(prediction)
        # lambda_val = lambda_dict['resale_price_lambda'] 
        # transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if data is not None else None
        # rounded_prediction = round(transformed_predict[0],2)
        # st.success(f"Based on the input, the Genie's price is,  {rounded_prediction:,.2f}")
        # st.info(f"On average, Genie's predictions are within approximately 10 to 20% of the actual market prices.")
        
elif selected == "Customer Insights":
    st.title("Customer Insights")
    if st.session_state.prediction is not None:
        st.write("K-means Prediction:", st.session_state.prediction['kmeans'])
        st.
        st.write("Classification Prediction:", st.session_state.prediction['classification'])
        st.write("Regression Prediction:", st.session_state.prediction['regression'])
    else:
        st.write("No prediction available. Please run a prediction first.")