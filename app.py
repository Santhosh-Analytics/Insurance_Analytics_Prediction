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
 



with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Customer Profile Input', 'Customer Insights'],
        icons=['house-door-fill', 'bar-chart-fill'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "12!important", "background-color": "gray"},
            "icon": {"color": "#000000", "font-size": "25px", "font-family": "Times New Roman"},
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

if selected == "Customer Profile Input":


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

        s = st.slider('Floor Area SQM:', min_value=20, max_value=500, value=65, help='Total Estimated space measured in square meters. Minimum value 20 sqm and maximum is 500 sqm.',)
#         floor = st.selectbox('Select number of floors:', floor_option, help="Estimated number of floors.")
#         if floor ==3:
#             floor_no_option = [number for number in range(3,52,3)]
#         else:
#             floor_no_option = [number for number in range(5,52,5)]
#         floor_level = st.selectbox('Select top floor level: ', floor_no_option, index=None, help="Estimated range of floors.", placeholder="Estimated range of floors.")
    
#         st.write(' ')
#         st.write(' ')
#         button = st.button('Predict Flat Price!')
    
    
#     remaining_lease_year = lease_year + 99 - date.today().year if lease_year is not None else None
#     floor_area_box = transform_single_value(floor_area, lambda_dict['floor_area_lambda'])     if floor_area is not None  else None
#     town_mapping={'Lim Chu Kang': 1, 'Queenstown': 2, 'Ang Mo Kio': 3, 'Clementi': 4, 'Geylang': 5, 'Bedok': 6, 'Bukit Batok': 7, 'Yishun': 8, 'Toa Payoh': 9, 'Jurong East': 10, 'Central Area': 11, 'Jurong West': 12, 'Kallang/Whampoa': 13, 'Woodlands': 14, 'Hougang': 15, 'Serangoon': 16, 'Marine Parade': 17, 'Bukit Merah': 18, 'Bukit Panjang': 19, 'Tampines': 20, 'Choa Chu Kang': 21, 'Sembawang': 22, 'Pasir Ris': 23, 'Bishan': 24, 'Bukit Timah': 25, 'Sengkang': 26, 'Punggol': 27}
#     year_mapping = {1990: 1, 1991: 2, 1992: 3, 1993: 4, 1994: 5, 1995: 6, 2002: 7, 2003: 8, 2004: 9, 2001: 10, 2005: 11, 2006: 12, 1999: 13, 2000: 14, 1998: 15, 1996: 16, 2007: 17, 1997: 18, 2008: 19, 2009: 20, 2010: 21, 2019: 22, 2015: 23, 2018: 24, 2011: 25, 2016: 26, 2017: 27, 2014: 28, 2020: 29, 2012: 30, 2013: 31, 2021: 32, 2022: 33, 2023: 34, 2024: 35}
#     flat_type_mapping = {'1 Room': 1, '2 Room': 2, '3 Room': 3, '4 Room': 4, '5 Room': 5, 'Executive': 6, 'Multi Generation': 7}
#     flat_model_mapping={'New Generation': 1, 'Standard': 2, 'Simplified': 3, 'Model A2': 4, '2-Room': 5, 'Model A': 6, 'Improved': 7, 'Improved-Maisonette': 8, 'Model A-Maisonette': 9, 'Premium Apartment': 10, 'Adjoined Flat': 11, 'Maisonette': 12, 'Apartment': 13, 'Terrace': 14, 'Multi Generation': 15, 'Premium Maisonette': 16, '3Gen': 17, 'Dbss': 18, 'Premium Apartment Loft': 19, 'Type S1': 20, 'Type S2': 21}
#     #lease_year_mapping={1969: 1, 1971: 2, 1967: 3, 1968: 4, 1973: 5, 1970: 6, 1972: 7, 1974: 8, 1977: 9, 1980: 10, 1983: 11, 1975: 12, 1981: 13, 1976: 14, 1978: 15, 1979: 16, 1966: 17, 1982: 18, 1985: 19, 1984: 20, 1986: 21, 1987: 22, 1988: 23, 1990: 24, 1989: 25, 1991: 26, 1997: 27, 1998: 28, 1996: 29, 1999: 30, 1994: 31, 1993: 32, 2000: 33, 1995: 34, 1992: 35, 2001: 36, 2002: 37, 2003: 38, 2004: 39, 2012: 40, 2014: 41, 2015: 42, 2005: 43, 2007: 44, 2010: 45, 2013: 46, 2008: 47, 2016: 48, 2009: 49, 2017: 50, 2018: 51, 2019: 52, 2006: 53, 2020: 54, 2011: 55}
#     # floor_mapping={1:0,2:.5,3: 1,4:1.5, 5: 2,6:2.5}
#     floor_level_mapping={3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 5: 6, 18: 7, 10: 8, 21: 9, 24: 10, 20: 11, 27: 12, 25: 13, 35: 14, 40: 15, 30: 16, 33: 17, 36: 18, 39: 19, 42: 20, 45: 21, 48: 22, 51: 23}
#     remaining_lease_year_mapping = {81: 1, 82: 2, 83: 3, 80: 4, 79: 5, 84: 6, 78: 7, 85: 8, 77: 9, 76: 10, 86: 11, 75: 12, 87: 13, 88: 14, 48: 15, 74: 16, 89: 17, 49: 18, 90: 19, 47: 20, 72: 21, 73: 22, 71: 23, 45: 24, 46: 25, 70: 26, 91: 27, 44: 28, 43: 29, 50: 30, 69: 31, 92: 32, 93: 33, 68: 34, 41: 35, 42: 36, 51: 37, 67: 38, 96: 39, 52: 40, 58: 41, 66: 42, 94: 43, 59: 44, 95: 45, 57: 46, 100: 47, 65: 48, 101: 49, 53: 50, 56: 51, 64: 52, 54: 53, 55: 54, 98: 55, 60: 56, 63: 57, 62: 58, 61: 59, 97: 60, 99: 61}
#     town_median_list = {1: 9.323219299316406,2: 9.260793685913086,3: 9.80174732208252,4: 10.193489074707031,5: 9.80174732208252,6: 10.354616165161133,7: 10.716485977172852,8: 10.615007400512695,9: 9.446247100830078,10: 10.866004943847656,11: 9.446247100830078,12: 11.342947006225586,13: 9.859149932861328,14: 11.29664134979248,15: 11.203139305114746,16: 10.963958740234375,17: 9.916055679321289,18: 9.916055679321289,19: 11.29664134979248,20: 11.342947006225586,21: 11.525309562683105,22: 11.250040054321289,23: 12.172235488891602,24: 11.388961791992188,25: 11.342947006225586,26: 11.203139305114746,27: 10.91515064239502}
    

    

#     town=town_mapping[town] if town is not None else None
#     year=year_mapping[date.today().year]
#     flat_type=flat_type_mapping[flat_type] if flat_type is not None else None
#     flat_model=flat_model_mapping[flat_model] if flat_model is not None else None
#    # lease_year=lease_year_mapping[lease_year] if lease_year is not None else None
#     # floor=floor_mapping[floor] if floor is not None else None
#     floor_level=floor_level_mapping[floor_level] if floor_level is not None else None
#     remaining_lease_year=remaining_lease_year_mapping[remaining_lease_year] if remaining_lease_year is not None else None
    
    
#     location_specifics = floor_area_box * town if None not in (floor_area_box, town) else None
#     #floor_area_year = floor_area_box / remaining_lease_yearj
    
#     age = year - lease_year if None not in (year, lease_year) else None
#     flat_area = flat_type * floor_area_box if None not in (flat_type, floor_area_box) else None
#     model_area = flat_model * floor_area_box if None not in (flat_model, floor_area_box) else None
#     town_mean_price = town_median_list[town] if None not in (town, town_median_list) else None
#     # st.write(lease_year)
#     floor_area_age = floor_area_box * (2024 - remaining_lease_year) if None not in (floor_area_box, lease_year) else None
#     floor_weightage = (floor_level -  floor ) * floor_area_box if None not in (floor_level, floor) else None
    
    # collisi1on_types = ["Front Collision", "Others", "Rear Collision", "Side Collision"]

# # Simulate user selection (you can replace this with a Streamlit selectbox)
# selected_collision = st.selectbox("Select Collision Type", collision_types)

# # Initialize an array with zeros
# one_hot_encoded_array = [0] * len(collision_types)

# # Get the index of the selected collision type
# selected_index = collision_types.index(selected_collision)

# # Set the value of the selected index to 1
# one_hot_encoded_array[selected_index] = 1

# # Convert the list to a comma-separated string
# one_hot_encoded_string = ', '.join(map(str, one_hot_encoded_array))

# # Display the one-hot encoded array as a string
# st.write("One-Hot Encoded Array:", one_hot_encoded_string)

    sex = {'Male': 1, 'Female':0}
    
    if education:
        encoded_edu=edu_pickle.transform([education])[0]
    
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
    
    st.write('encoded_edu')
    
J    st.write(data)
    scaled_data = scale_reg.transform(data)
    # st.write(scaled_data)
    
    

    if button and data is not None:
        
        scaled_data = scale_reg.transform(data)
        # st.write(scaled_data)
        prediction = XGB_model.predict(scaled_data)
        # st.write(prediction)
        lambda_val = lambda_dict['resale_price_lambda'] 
        transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if data is not None else None
        rounded_prediction = round(transformed_predict[0],2)
        st.success(f"Based on the input, the Genie's price is,  {rounded_prediction:,.2f}")
        st.info(f"On average, Genie's predictions are within approximately 10 to 20% of the actual market prices.")
        
