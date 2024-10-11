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
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler,LabelEncoder

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
    if value is None or not isinstance(value, (int, float)):
        return None  # Handle missing value
    elif value <= 0:
        raise ValueError("All fields require positive input values.")
    transformed_value = boxcox([value + np.spacing(1)], lmbda=lmbda)[0]
    return transformed_value


def reverse_boxcox_transform(predicted, lambda_val):
    return inv_boxcox(predicted, lambda_val)

# Load the saved lambda values
with open(r'pkls/transformations/transformation_params.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

with open(r'pkls/edu_en.pkl', 'rb') as f:
    edu_pickle = pickle.load(f)

    
with open(r'pkls/scale_reg.pkl', 'rb') as f:
    scale_reg = pickle.load(f)
    
with open(r'pkls/K_means.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open(r'pkls/ADA_Classifier.pkl', 'rb') as f:
    Class_model = pickle.load(f)
    

with open(r'pkls/GB_Regressor.pkl', 'rb') as f:
    Reg_model = pickle.load(f)
    

    
    
    
with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Customer Insights and Predictions', 'Customer Insights'],
        icons=['house-door-fill', 'bar-chart-fill'],
        menu_icon="cast",
        default_index=0,
        key='menu_option',
        styles={
            "container": {"padding": "12!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px", "font-family": "Roboto Condensed"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "#ffffff", "color": "#55ACEE", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)


    
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

if selected == "Customer Insights and Predictions":
    st.title("Customer Profile")

    selected2 = option_menu(None, ["Customer Characters", "Fraud Detection", "Claim Amount Prediction"], 
    icons=['house', 'cloud-upload', "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
        )

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
    hobbies_opt = sorted(hobbies_opt)
    insured_opt = ['husband', 'not-in-family', 'other-relative','own-child',   'unmarried','wife']
    incident_opt = ['Multi-vehicle Collision', 'Parked Car', 'Single Vehicle Collision' ,'Vehicle Theft', ]
    make_opt = ['Dodge', 'Suburu', 'Saab', 'Nissan', 'Chevrolet', 'BMW', 'Ford', 'Toyota', 'Audi', 'Accura',
         'Volkswagen', 'Jeep', 'Mercedes', 'Honda']
    collision_opt = ['Front Collision','Others','Rear Collision','Side Collision']
    severity_opt = ['Trivial Damage', 'Major Damage','Minor Damage','Total Loss']
    auth_opt = ['Police','Other','Fire','Ambulance']
    city_opt = ['Arlington', 'Columbus', 'Hillsdale', 'Northbend', 'Northbrook', 'Riverwood', 'Springfield']
    hour_opt = [17,  3,  0, 23, 16, 10,  4, 13,  6, 14,  9, 21, 18, 19, 12,
                   7, 15, 22,  8, 20,  5,  2, 11,  1]
    vehicle_opt = [1, 2, 3, 4]
    prpty_dmg_opt = ['YES', 'NO']
    injury_opt = [0, 1, 2]
    wit_opt = [0, 1, 2, 3]
    fir_opt = ['NO', 'YES']
    
    
    col1, col, col2 = st.columns([2,.5,2])
    

    with col1:
        cust_month = st.number_input('Enter the Customer tenure ranges:', help="Enter the Customer tenure ranges. If new customer enter 0:",step = 1)
        policy_state = st.selectbox('Select policy State:', states    ,  help="Select Policy State/Location" )
        policy_deduc = st.selectbox('Select deductable:', policy_deduc_opt,  help="Portion of a claim that policy holder responsible to pay." )
        policy_premium = st.number_input('Enter annual premium amount:', help="Enter annual premium amount",step = 100)
        st.write('Policy Premium:', policy_premium)

        vehi_claim_amount = st.number_input('Enter vehicle claim amount:', help="Enter vehicle claim amount",step = 100)
        st.write('Vehicle Claim Amount:', vehi_claim_amount)

        cust_age = st.number_input('Enter the Customer age:', help="Enter the Customer age:",step=1)
        st.write('Customer Age:', cust_age)

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
        incident_severity = st.selectbox('Select Incident severity:', severity_opt, help="Select severity type")
        auth = st.selectbox('Authority Contacted:', auth_opt, help="Has any goverment authority contacted?")
        city = st.selectbox('Incident City:', city_opt, help="City where the incodent occured.")
        hour = st.selectbox('Incident Time:', hour_opt, help="Time when the incodent occured.")
        no_of_veh = st.selectbox('No of Vehicle Involved:', vehicle_opt, help="Vehicles count that met with an incident.")
        prpty_dmg = st.selectbox('Property Damage:', prpty_dmg_opt, help="Any property damaged due to the incident.")
        injury = st.selectbox('Injury:', injury_opt, help="No of people injured.")
        wit = st.selectbox('No of witness:', wit_opt, help="No of witness for the incident.")
        fir = st.selectbox('Police Report:', fir_opt, help="Reported to police.")

       
        st.write(' ')
        st.write(' ')
        button = st.button('Get Insights!')
    
    

    sex = {'Male': 1, 'Female':0}
    
    if education:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoded_edu = edu_pickle.transform([education])[0]
        except Exception as e:
            st.error(f"Error encoding education: {e}")
            encoded_edu = None  
    
    coll_array = [0] * len(collision_opt)
    selected_index = collision_opt.index(collision_type)
    coll_array[selected_index]= 1
   
    
    inc_array = [0] * len(incident_opt)
    selected_inc_index = incident_opt.index(incident_type)
    inc_array[selected_inc_index]= 1
    
    
    rela_array = [0] * len(insured_opt)
    selected_rela_index = insured_opt.index(insured)
    rela_array[selected_rela_index]= 1
    
    
    hobbies_array = [0] * len(hobbies_opt)
    selected_hoobies_index = hobbies_opt.index(hobbies)
    hobbies_array[selected_hoobies_index]= 1
    
    
    
    age_box = transform_single_value(cust_age, lambda_dict.get('age_boxcox')) if cust_age and cust_age > 0 else None
    policy_premium_box = transform_single_value(policy_premium, lambda_dict.get('policy_annual_premium_boxcox')) if policy_premium and policy_premium > 0  else None
    vehicle_claim_box = transform_single_value(vehi_claim_amount, lambda_dict.get('vehicle_claim_boxcox')) if vehi_claim_amount and vehi_claim_amount >0  else None
    

    
    data_clus = np.array([[cust_month,policy_deduc,sex[insured_sex],year,encoded_edu, age_box]  + coll_array + hobbies_array + rela_array + inc_array ])
    st.write('Clustering Data:','\n',data_clus)
    
    data_reg = np.array([[incident_severity, collision_type, policy_premium_box, cust_month, age_box, auto_make] + 
                         hobbies_array +  [occupation, vehicle_claim_box] ])
    st.write('Regression Data:','\n',data_reg)
    
    
    
    
    # fea2 = ['incident_severity', 'collision_type','policy_annual_premium_boxcox', 'months_as_customer','age_boxcox',
    #     'insurance_age','auto_make','insured_hobbies','insured_occupation','vehicle_claim_boxcox',]
    
    # Continuous columns: ['age_boxcox', 'policy_annual_premium_boxcox', 'insurance_age', 'vehicle_age', 'auto_year']
    # Nominal columns: ['policy_state', 'insured_sex', 'insured_occupation', 'insured_hobbies', 'insured_relationship', 'incident_type', 'collision_type', 'authorities_contacted', 'incident_state', 'incident_city', 'property_damage', 'police_report_available', 'auto_make', 'fraud_reported']
    
    # st.write(scale_reg)
    # scaled_data_reg = scale_reg.transform(data_reg)
    # st.write(scaled_data_reg)
    
    
    # st.session_state.data_clus = data_clus
    # st.session_state.data_reg = data_reg

    # if button:
        # if st.session_state.data_clus is not None and st.session_state.data_reg is not None:
            # try:
                # Your prediction logic here
                # kmeans_prediction = kmeans.predict(st.session_state.data_clus)
                # class_prediction = Class_model.predict(st.session_state.data_clus)
                # reg_prediction = Reg_model.predict(st.session_state.data_clus)
                
                # Store predictions in session state
                # st.session_state.prediction = {
                #     'kmeans': kmeans_prediction,
                    # 'classification': class_prediction,
                    # 'regression': reg_prediction
                # }
                
                # Set flag to navigate to Customer Insights
    #             st.session_state.current_page = "Customer Insights"
    #             st.experimental_rerun()
    #         except Exception as e:
    #             st.error(f"Error: {e}")
    # else:
    #     st.warning("Please enter customer data before running the prediction.")
                
        # st.write(prediction)
        # lambda_val = lambda_dict['resale_price_lambda'] 
        # transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if data is not None else None
        # rounded_prediction = round(transformed_predict[0],2)
        # st.success(f"Based on the input, the Genie's price is,  {rounded_prediction:,.2f}")
        # st.info(f"On average, Genie's predictions are within approximately 10 to 20% of the actual market prices.")
    # else:
    st.write('End of profile')
elif selected == "Customer Insights":
    
    st.markdown("# <span style='color:blue;'>Customer Insights:</span>", unsafe_allow_html=True)

    st.markdown("""<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: inherit;text-indent: 4em;'>
         In this section, we will explore customer characteristics and behavior, tailored marketing strategies, product recommendations, 
         cross-selling opportunities, and engagement strategies, all based on the input from customer profile details.

</p>""", unsafe_allow_html=True)
    
    # if st.button("Go Back"):
        # st.session_state.current_page = "Customer Profile Input"
        # st.experimental_rerun()
    
    selected2 = option_menu(None, ["Segment Overview", "Fraud Detection", "Claim Amount Prediction"], 
    icons=['house', 'cloud-upload', "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
        )
    
    if selected2 == 'Segment Overview':
        st.markdown("## <span style='color:blue;'>Customer Segment Overview:</span>", unsafe_allow_html=True)
        if st.button('Test'):
            selected2 == "Fraud Detection"
        
        if 'prediction' in st.session_state and st.session_state.prediction is not None and 'kmeans' in st.session_state.prediction:
            # kmeans_prediction = st.session_state.prediction['kmeans']
            st.write("K-means Prediction:", st.session_state.prediction['kmeans'])
            if st.session_state.prediction['kmeans'] == 0:
                st.write("K-means Prediction:", st.session_state.prediction['kmeans'])
                st.image('Cluster_0.png',use_column_width=True)            
            
            elif 'prediction' in st.session_state and st.session_state.prediction['kmeans'] == 1:
                st.image('Cluter_1.png',use_column_width=True)
                
            elif 'prediction' in st.session_state and st.session_state.prediction['kmeans'] == 2:
                st.image('Cluster_2.png',use_column_width=True)
            
        else:
            st.info("No prediction available. Please update data in the 'Cutomer Profile Input' and hit 'Get Insights'.")

    
    
    
    if st.session_state.prediction is not None:
        st.write("K-means Prediction:", st.session_state.prediction['kmeans'])
        # st.write("Classification Prediction:", st.session_state.prediction['classification'])
        # st.write("Regression Prediction:", st.session_state.prediction['regression'])
    else:
        st.info("No prediction available. Please update data in the 'Cutomer Profile Input' and hit 'Get Insights'.")