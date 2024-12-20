import pickle
import warnings
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import stats
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, RobustScaler,
                                   StandardScaler)
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Insurance Tailorings",
    page_icon=r"insurance.jpeg",
)


page_bg_img = """
<style>
    .stApp {
        # background-image: url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fe1.pxfuel.com%2Fdesktop-wallpaper%2F246%2F629%2Fdesktop-wallpaper-glossy-black-shiny-black.jpg&f=1&nofb=1&ipt=94aa5d1f9d650cac934f99411cce8fa55ef43d1aabce3f3faffc7ed1776c144e&ipo=images");
        background-image:url("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwallpaperset.com%2Fw%2Ffull%2Fc%2F4%2F2%2F2932.jpg&f=1&nofb=1&ipt=167b8ede7fb9039711212e5ae543004922bc451239b539ed9d447699ccab439d&ipo=images")
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Injecting CSS for custom styling for button
st.markdown(
    """
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
        background-color: darkgreen;
        color: white;
        border: single;
        border-width: 5px;
        border-color: #3e8e41;
        padding: 10px 20px;
        text-align: center;
        text-decoration: text-overflow;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 22px;
    }
    .stButton>button:hover {
        background-color: forestgreen !important;
        color: white !important;
        border: signle !important;
        border-width: 5px !important;
        border-color: Darkgreen !important;
        font-size: 22px !important;
        text-decoration: text-overflow !important;
        transition: width 2s !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Injecting custom styling to display info

st.markdown(
    """
    <style>
    .custom-info-box {
        background-color: rgba(61, 157, 243, 0.2);  /* Background color similar to st.info */
        padding: 10px;  /* Add some padding for spacing */
        border-left: 10px solid #1E88E5;  /* Add a colored border on the left */
        border-right: 10px solid #1E88E5;
        border-up: 10px solid #1E88E5;
        border-down: 10px solid #1E88E5;
        border-radius: 25px;  /* Rounded corners */
        font-family: Arial, sans-serif;  /* Font styling */
        font-size: 25px;  /* Font size adjustment */
        color: #ffffff;  /* Font color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

texts = """
<div class="custom-info-box">
    <strong>Information:</strong> Please update customer information to proceed.
</div>
"""


# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None or not isinstance(value, (int, float)):
        return None  # Handle missing value
    elif value <= 0:
        raise ValueError("All fields require positive input values.")
    transformed_value = boxcox([value + np.spacing(1)], lmbda=lmbda)[0]
    return transformed_value


def reverse_boxcox_transform(predicted, lambda_val):
    # Convert predicted values to a numpy array to ensure compatibility
    predicted = np.array(predicted, dtype=float).flatten()  # Flatten the array
    if len(predicted) == 1:
        predicted = predicted[0]  # If it's a single value, extract it
    return inv_boxcox(predicted, lambda_val)


# Load the saved lambda values
with open(r"pkls/transformations/transformation_params.pkl", "rb") as f:
    lambda_dict = pickle.load(f)

with open(r"pkls/edu_en.pkl", "rb") as f:
    edu_pickle = pickle.load(f)


# with open(r'pkls/scale_reg.pkl', 'rb') as f:
#     scale_reg = pickle.load(f)

with open(r"pkls/K_means.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open(r"pkls/ADA_Classifier.pkl", "rb") as f:
    Class_model = pickle.load(f)


# with open(r"pkls/GB_Regressor1.pkl", "rb") as f:
#     Reg_model = joblib.load(f)
Reg_model = joblib.load(r"pkls/GB_Regressor1.pkl")


with open(f"pkls/Reg_EN/encoder_collision_type.pkl", "rb") as f:
    coll_type_en = pickle.load(f)

with open(f"pkls/Reg_EN/encoder_auto_make.pkl", "rb") as f:
    auto_make_en = pickle.load(f)

with open(f"pkls/Reg_EN/encoder_insured_hobbies.pkl", "rb") as f:
    hobbies_en = pickle.load(f)


with open(f"pkls/Reg_EN/encoder_insured_occupation.pkl", "rb") as f:
    occu_en = pickle.load(f)


with open(f"pkls/en_class/edu_class.pkl", "rb") as f:
    edu_en_class = pickle.load(f)


with open(f"pkls/en_class/severity_class.pkl", "rb") as f:
    severity_en_class = pickle.load(f)


with open(f"pkls/scaling/class_scale.pkl", "rb") as f:
    class_scale = pickle.load(f)


def encode_feature(value, encoder, feature_name):
    # Create a single-element DataFrame with the correct column name
    df = pd.DataFrame({feature_name: [value]})
    # Transform the DataFrame
    encoded_df = encoder.transform(df)
    # Return the encoded value
    return encoded_df.iloc[0, 0]


def encode_months_as_customer(months):
    en_list = {
        410: 1,
        419: 2,
        303: 3,
        59: 4,
        389: 5,
        17: 6,
        213: 7,
        77: 8,
        98: 9,
        35: 10,
        237: 11,
        324: 12,
        425: 13,
        157: 14,
        167: 15,
        90: 16,
        153: 17,
        415: 18,
        72: 19,
        315: 20,
        10: 21,
        385: 22,
        227: 23,
        236: 24,
        467: 25,
        62: 26,
        472: 27,
        202: 28,
        291: 29,
        399: 30,
        465: 31,
        109: 32,
        468: 33,
        317: 34,
        308: 35,
        270: 36,
        66: 37,
        463: 38,
        5: 39,
        81: 40,
        131: 41,
        356: 42,
        149: 43,
        412: 44,
        162: 45,
        288: 46,
        1: 47,
        94: 48,
        87: 49,
        478: 50,
        404: 51,
        152: 52,
        241: 53,
        83: 54,
        38: 55,
        284: 56,
        231: 57,
        6: 58,
        136: 59,
        258: 60,
        61: 61,
        32: 62,
        386: 63,
        193: 64,
        456: 65,
        243: 66,
        405: 67,
        332: 68,
        101: 69,
        113: 70,
        426: 71,
        287: 72,
        435: 73,
        104: 74,
        121: 75,
        132: 76,
        440: 77,
        392: 78,
        334: 79,
        230: 80,
        189: 81,
        108: 82,
        36: 83,
        406: 84,
        119: 85,
        107: 86,
        438: 87,
        112: 88,
        390: 89,
        100: 90,
        464: 91,
        26: 92,
        429: 93,
        473: 94,
        161: 95,
        461: 96,
        447: 97,
        125: 98,
        398: 99,
        130: 100,
        57: 101,
        12: 102,
        175: 103,
        199: 104,
        73: 105,
        362: 106,
        255: 107,
        133: 108,
        129: 109,
        3: 110,
        247: 111,
        103: 112,
        16: 113,
        208: 114,
        95: 115,
        128: 116,
        69: 117,
        8: 118,
        239: 119,
        106: 120,
        252: 121,
        261: 122,
        159: 123,
        427: 124,
        293: 125,
        344: 126,
        120: 127,
        296: 128,
        370: 129,
        310: 130,
        439: 131,
        85: 132,
        91: 133,
        50: 134,
        9: 135,
        126: 136,
        409: 137,
        350: 138,
        164: 139,
        290: 140,
        99: 141,
        245: 142,
        200: 143,
        105: 144,
        55: 145,
        257: 146,
        174: 147,
        47: 148,
        253: 149,
        232: 150,
        27: 151,
        22: 152,
        337: 153,
        156: 154,
        118: 155,
        180: 156,
        178: 157,
        140: 158,
        224: 159,
        312: 160,
        278: 161,
        446: 162,
        210: 163,
        60: 164,
        322: 165,
        218: 166,
        289: 167,
        165: 168,
        168: 169,
        41: 170,
        138: 171,
        298: 172,
        437: 173,
        182: 174,
        251: 175,
        190: 176,
        260: 177,
        97: 178,
        89: 179,
        211: 180,
        155: 181,
        70: 182,
        475: 183,
        75: 184,
        235: 185,
        48: 186,
        330: 187,
        479: 188,
        40: 189,
        33: 190,
        407: 191,
        379: 192,
        343: 193,
        219: 194,
        316: 195,
        45: 196,
        209: 197,
        222: 198,
        19: 199,
        54: 200,
        124: 201,
        146: 202,
        242: 203,
        163: 204,
        266: 205,
        154: 206,
        431: 207,
        117: 208,
        2: 209,
        364: 210,
        216: 211,
        217: 212,
        413: 213,
        39: 214,
        441: 215,
        286: 216,
        355: 217,
        173: 218,
        375: 219,
        194: 220,
        171: 221,
        272: 222,
        335: 223,
        267: 224,
        250: 225,
        78: 226,
        220: 227,
        46: 228,
        402: 229,
        80: 230,
        283: 231,
        145: 232,
        31: 233,
        186: 234,
        263: 235,
        116: 236,
        277: 237,
        273: 238,
        137: 239,
        147: 240,
        294: 241,
        453: 242,
        264: 243,
        127: 244,
        276: 245,
        338: 246,
        271: 247,
        212: 248,
        244: 249,
        102: 250,
        319: 251,
        225: 252,
        229: 253,
        192: 254,
        256: 255,
        201: 256,
        88: 257,
        259: 258,
        269: 259,
        30: 260,
        93: 261,
        151: 262,
        187: 263,
        207: 264,
        197: 265,
        64: 266,
        430: 267,
        234: 268,
        4: 269,
        143: 270,
        249: 271,
        414: 272,
        198: 273,
        360: 274,
        185: 275,
        172: 276,
        179: 277,
        285: 278,
        7: 279,
        169: 280,
        11: 281,
        321: 282,
        214: 283,
        150: 284,
        305: 285,
        292: 286,
        148: 287,
        111: 288,
        396: 289,
        195: 290,
        65: 291,
        114: 292,
        295: 293,
        359: 294,
        274: 295,
        372: 296,
        380: 297,
        436: 298,
        279: 299,
        280: 300,
        342: 301,
        448: 302,
        0: 303,
        460: 304,
        166: 305,
        265: 306,
        275: 307,
        82: 308,
        206: 309,
        205: 310,
        223: 311,
        328: 312,
        134: 313,
        203: 314,
        122: 315,
        188: 316,
        394: 317,
        191: 318,
        177: 319,
        115: 320,
        228: 321,
        313: 322,
        123: 323,
        160: 324,
        371: 325,
        84: 326,
        434: 327,
        34: 328,
        233: 329,
        142: 330,
        51: 331,
        254: 332,
        184: 333,
        449: 334,
        300: 335,
        428: 336,
        96: 337,
        458: 338,
        381: 339,
        204: 340,
        43: 341,
        86: 342,
        56: 343,
        297: 344,
        58: 345,
        110: 346,
        282: 347,
        347: 348,
        246: 349,
        215: 350,
        238: 351,
        135: 352,
        281: 353,
        325: 354,
        369: 355,
        304: 356,
        354: 357,
        144: 358,
        309: 359,
        20: 360,
        240: 361,
        37: 362,
        158: 363,
        29: 364,
        25: 365,
        183: 366,
        299: 367,
        422: 368,
        421: 369,
        13: 370,
        76: 371,
        63: 372,
        366: 373,
        476: 374,
        196: 375,
        14: 376,
        92: 377,
        411: 378,
        141: 379,
        373: 380,
        176: 381,
        451: 382,
        339: 383,
        79: 384,
        377: 385,
        15: 386,
        53: 387,
        320: 388,
        352: 389,
        67: 390,
        24: 391,
    }

    if months in en_list:
        return en_list[months]
    else:
        # Find the closest key in en_list
        closest_key = min(en_list.keys(), key=lambda x: abs(x - months))
        return en_list[closest_key]


with st.sidebar:
    st.markdown("<hr style='border: 4px solid #ffffff;'>", unsafe_allow_html=True)

    selected = option_menu(
        "Main Menu",
        ["About", "Customer Insights and Predictions"],
        icons=["house-door-fill", "bar-chart-fill"],
        menu_icon="cast",
        default_index=0,
        key="menu_option",
        styles={
            "container": {
                "padding": "1 !important",
                "background-color": "black",
                "border-radius": "15px",
                "font": "JetBrainsMono Nerd Font",
                "border": "1px solid grey",
            },
            "icon": {
                "color": "orange",
                "font-size": "25px",
                "font-family": "JetBrainsMono Nerd Font",
            },
            "nav-link": {
                "font-size": "20px",
                "color": "#ffffff",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#000000ee",
                "box-shadow": "0 2px 4px Coral",
                "font": "JetBrainsMono Nerd Font",
                "border": "1px solid Coral",
                "cursor": "pointer",
                "transition": "background-color 0.5s ease, color 0.5s ease",
                "text-shadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
            },
            "nav-link-selected": {
                "background-color": "Coral",
                "border-radius": "25px",
                "font": "JetBrainsMono Nerd Font",
                "border": "0px solid #ddd",
            },
        },
    )
    st.markdown("<hr style='border: 4px solid #ffffff;'>", unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align: center; font-size: 38px; color: #ffffff; font-weight: 700; font-family: JetBrainsMono Nerd Font;'>Insurance Analytics & Insights</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; font-size: 26px; color: #ffffff; font-weight: 500; font-family: JetBrainsMono Nerd Font;'>Data-Driven Solutions for Customer Segmentation and Fraud Detection</h4>",
    unsafe_allow_html=True,
)

st.markdown("<hr style='border: 4px solid beige;'>", unsafe_allow_html=True)


if selected == "About":
    st.markdown(
        "<h3 style='text-align: center; font-size: 38px; color: Aquamarine; font-weight: 700; font-family: JetBrainsMono Nerd Font;'>Understanding Insurance Insights</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='font-size: 30px; text-align: left; font-family: JetBrainsMono Nerd Font; color: #da532c;'> Overview </h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 4em;'>
         The objective of this app is to leverage trained machine learning alogrithm to extract valuable insights from insurance data.
        By utilizing predictive models, this app provides data-driven insights to help insurance companies make Enhancing Decision-Making,
        Optimizing Risk Assessment, Improving Operational Efficiency.

</p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='font-size: 34px; text-align: left; font-family: JetBrainsMono Nerd Font; color: Aquamarine;'> Instructions to use this app: </h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='font-size: 30px; text-align: left; font-family: JetBrainsMono Nerd Font; color: #da532c;'> Customer Profile Input: </h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 4em;'>
        Enter customer details such as age, gender, and policy information. The app will generate personalized segmentation and predictions.
</p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='font-size: 30px; text-align: left; font-family: JetBrainsMono Nerd Font; color: #da532c;'> Customer Insights: </h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 4em;'>
        See detailed customer segmentation, marketing strategies, risk profiles, and predictions for fraud detection and premium pricing based on the input data.
    </p>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 4em;'>
        Whether you're an underwriter, risk manager, or marketer, this app provides actionable insights to help you make data-driven decisions that benefit both your business and your customers.
    </p>""",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border: 4px solid #ffffff;'>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='font-size: 30px; text-align: left; font-family: JetBrainsMono Nerd Font; color: #da532c;'> Contributing </h3>",
        unsafe_allow_html=True,
    )
    github_url = (
        "https://github.com/Santhosh-Analytics/Singapore-Resale-Flat-Prices-Predicting"
    )
    st.markdown(
        """<p style='text-align: left; font-size: 22px;text-indent: 4em; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;'>
        Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
    </p>""".format(
            github_url
        ),
        unsafe_allow_html=True,
    )

if selected == "Customer Insights and Predictions":
    selected2 = option_menu(
        None,
        ["Customer Characteristics", "Fraud Detection", "Claim Amount Prediction"],
        icons=["house", "cloud-upload", "list-task"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "darkgreen",
                "border-radius": "10px",
                "font": "JetBrainsMono Nerd Font",
                "border": "1px solid #ddd",
                "text-align": "center",
            },
            "icon": {
                "color": "black",
                "font-size": "25px",
            },
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
                "box-shadow": "0 4px 8px #ddd",
                "font": "JetBrainsMono Nerd Font",
                "border": "1px solid #ddd",
                "color": "black",
                "cursor": "pointer",
                "transition": "background-color 0.3s ease, color 0.3s ease",
                "text-align": "center",
            },
            "nav-link-selected": {
                "background-color": "Chocolate",
                "border-radius": "25px",
                "box-shadow": "0 4px 8px #ddd",
                "font": "JetBrainsMono Nerd Font",
                "border": "1px solid #ddd",
            },
        },
    )

    st.markdown(
        "<h4 style='text-align: center; font-size: 36px; color: Aquamarine; font-weight: 500; font-family: JetBrainsMono Nerd Font;'>Customer Profile</h4>",
        unsafe_allow_html=True,
    )

    # Options for various dropdowns
    states = ("OH", "IL", "IN")
    policy_deduc_opt = (500, 1000, 1500, 2000)
    edu_opt = ("JD", "High School", "Associate", "MD", "Masters", "PhD", "College")
    occu_opt = (
        "machine-op-inspct",
        "prof-specialty",
        "tech-support",
        "sales",
        "exec-managerial",
        "craft-repair",
        "transport-moving",
        "other-service",
        "priv-house-serv",
        "armed-forces",
        "adm-clerical",
        "protective-serv",
        "handlers-cleaners",
        "farming-fishing",
    )
    hobbies_opt = [
        "reading",
        "exercise",
        "paintball",
        "bungie-jumping",
        "golf",
        "movies",
        "camping",
        "kayaking",
        "yachting",
        "hiking",
        "video-games",
        "base-jumping",
        "skydiving",
        "board-games",
        "polo",
        "chess",
        "dancing",
        "sleeping",
        "cross-fit",
        "basketball",
    ]
    hobbies_opt = sorted(hobbies_opt)
    insured_opt = [
        "husband",
        "not-in-family",
        "other-relative",
        "own-child",
        "unmarried",
        "wife",
    ]
    incident_opt = [
        "Multi-vehicle Collision",
        "Parked Car",
        "Single Vehicle Collision",
        "Vehicle Theft",
    ]
    make_opt = [
        "Dodge",
        "Suburu",
        "Saab",
        "Nissan",
        "Chevrolet",
        "BMW",
        "Ford",
        "Toyota",
        "Audi",
        "Accura",
        "Volkswagen",
        "Jeep",
        "Mercedes",
        "Honda",
    ]
    collision_opt = ["Front Collision", "Others", "Rear Collision", "Side Collision"]
    severity_opt = ["Trivial Damage", "Major Damage", "Minor Damage", "Total Loss"]
    auth_opt = ["Police", "Other", "Fire", "Ambulance"]
    city_opt = [
        "Arlington",
        "Columbus",
        "Hillsdale",
        "Northbend",
        "Northbrook",
        "Riverwood",
        "Springfield",
    ]
    hour_opt = [
        17,
        3,
        0,
        23,
        16,
        10,
        4,
        13,
        6,
        14,
        9,
        21,
        18,
        19,
        12,
        7,
        15,
        22,
        8,
        20,
        5,
        2,
        11,
        1,
    ]
    vehicle_opt = [1, 2, 3, 4]
    prpty_dmg_opt = ["YES", "NO"]
    injury_opt = [0, 1, 2]
    wit_opt = [0, 1, 2, 3]
    fir_opt = ["NO", "YES"]
    inc_state_opt = ["NC", "NY", "OH", "PA", "SC", "VA", "WV"]

    # col1, col, col2 = st.columns([2,.5,2])

    # with col1:
    #     cust_month = st.number_input('Enter the Customer tenure ranges:', help="Enter the Customer tenure ranges. If new customer enter 0:",step = 1)
    #     policy_state = st.selectbox('Select policy State:', states    ,  help="Select Policy State/Location" )
    #     policy_deduc = st.selectbox('Select deductable:', policy_deduc_opt,  help="Portion of a claim that policy holder responsible to pay." )
    #     policy_premium = st.number_input('Enter annual premium amount:', help="Enter annual premium amount",step = 100)
    #     st.write('Policy Premium:', policy_premium)

    #     vehi_claim_amount = st.number_input('Enter vehicle claim amount:', help="Enter vehicle claim amount",step = 100)
    #     st.write('Vehicle Claim Amount:', vehi_claim_amount)

    #     cust_age = st.number_input('Enter the Customer age:', help="Enter the Customer age:",step=1)
    #     st.write('Customer Age:', cust_age)

    #     insured_sex = st.selectbox('Select gender:', ['Male', 'Female'],  help="Customer Gender")
    #     education = st.selectbox('Select education:', edu_opt, help="Customer education Level")
    #     occupation = st.selectbox('Select occupation:', occu_opt,  help="Customer occupation")
    #     hobbies = st.selectbox('Select hobbies:', hobbies_opt,  help="Select hobbies")
    #     insured = st.selectbox('Select insured relation:', insured_opt,  help="Select insured relation")
    #     fraud = st.selectbox('Select Fraud :', [True,False],  help="Select insured relation")

    # with col2:
    #     auto_make = st.selectbox('Select auto make:', make_opt, help="Select Vehicle make")
    #     year = st.selectbox('Select make year:', [i for i in range(1994, 2016)],  help="Select Vehicle make year")
    #     incident_type = st.selectbox('Select incident type:', incident_opt,  help="Select incident type")
    #     collision_type = st.selectbox('Select Collision type:', collision_opt, help="Select Collision type")
    #     incident_severity = st.selectbox('Select Incident severity:', severity_opt, help="Select severity type")
    #     auth = st.selectbox('Authority Contacted:', auth_opt, help="Has any goverment authority contacted?")
    #     city = st.selectbox('Incident City:', city_opt, help="City where the incodent occured.")
    #     hour = st.selectbox('Incident Time:', hour_opt, help="Time when the incodent occured.")
    #     no_of_veh = st.selectbox('No of Vehicle Involved:', vehicle_opt, help="Vehicles count that met with an incident.")
    #     prpty_dmg = st.selectbox('Property Damage:', prpty_dmg_opt, help="Any property damaged due to the incident.")
    #     injury = st.selectbox('Injury:', injury_opt, help="No of people injured.")
    #     wit = st.selectbox('No of witness:', wit_opt, help="No of witness for the incident.")
    #     fir = st.selectbox('Police Report:', fir_opt, help="Reported to police.")

    if selected2 == "Customer Characteristics":
        col1, col, col2 = st.columns([2, 0.5, 2])
        with col1:
            insured_sex = st.selectbox(
                "Select gender:", ["Male", "Female"], help="Customer Gender"
            )
            education = st.selectbox(
                "Select education:", edu_opt, help="Customer education Level"
            )
            cust_age = st.number_input(
                "Enter the Customer age:", help="Enter the Customer age:", step=1
            )
            hobbies = st.selectbox(
                "Select hobbies:", hobbies_opt, help="Select hobbies"
            )
            insured = st.selectbox(
                "Select insured relation:", insured_opt, help="Select insured relation"
            )

        with col2:
            cust_month = st.number_input(
                "Enter the Customer tenure ranges:",
                help="Enter the Customer tenure ranges. If new customer enter 0:",
                step=1,
            )
            policy_deduc = st.selectbox(
                "Select deductable:",
                policy_deduc_opt,
                help="Portion of a claim that policy holder responsible to pay.",
            )
            year = st.selectbox(
                "Select make year:",
                [i for i in range(1994, 2016)],
                help="Select Vehicle make year",
            )
            collision_type = st.selectbox(
                "Select Collision type:", collision_opt, help="Select Collision type"
            )
            incident_type = st.selectbox(
                "Select incident type:", incident_opt, help="Select incident type"
            )

        sex = {"Male": 1, "Female": 0}

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
        coll_array[selected_index] = 1

        inc_array = [0] * len(incident_opt)
        selected_inc_index = incident_opt.index(incident_type)
        inc_array[selected_inc_index] = 1

        rela_array = [0] * len(insured_opt)
        selected_rela_index = insured_opt.index(insured)
        rela_array[selected_rela_index] = 1

        hobbies_array = [0] * len(hobbies_opt)
        selected_hoobies_index = hobbies_opt.index(hobbies)
        hobbies_array[selected_hoobies_index] = 1

        age_box = (
            transform_single_value(cust_age, lambda_dict.get("age_boxcox"))
            if cust_age and cust_age > 0
            else None
        )

        button = (
            st.button("Get Insights!")
            if (
                cust_age is not None
                and cust_month is not None
                and cust_age > 0
                and cust_month > 0
            )
            else st.markdown(texts, unsafe_allow_html=True)
        )
        preds = 0

        data_clus = np.array(
            [
                [cust_month, policy_deduc, sex[insured_sex], year, encoded_edu, age_box]
                + coll_array
                + hobbies_array
                + rela_array
                + inc_array
            ]
        )
        # st.write('Clustering Data:','\n',data_clus)

        if (
            cust_age is not None
            and cust_month is not None
            and cust_age > 0
            and cust_month > 0
            and button
        ):
            preds = kmeans.predict(data_clus)
            # st.write(preds)
        st.markdown("<hr style='border: 4px solid #ffffff;'>", unsafe_allow_html=True)
        if button == 1:
            st.markdown(
                "# <span style='color:Aquamarine;'>Customer Insights:</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 4em;'>
         In this section, we will explore customer characteristics and behavior, tailored marketing strategies, product recommendations, 
         cross-selling opportunities, and engagement strategies, all based on the input from customer profile details. </p>""",
                unsafe_allow_html=True,
            )

            st.markdown(
                "## <span style='color:#da532c;'>Customer Segment Overview:</span>",
                unsafe_allow_html=True,
            )

            if preds == 0:
                st.image("Cluster_0.png", use_column_width=True)

            elif preds == 1:
                st.image("Cluter_1.png", use_column_width=True)

            elif preds == 2:
                st.image("Cluster_2.png", use_column_width=True)

    elif selected2 == "Claim Amount Prediction":
        col1, col, col2 = st.columns([2, 0.5, 2])

        with col1:
            collision_type = st.selectbox(
                "Select Collision type:",
                collision_opt,
                help="Select Collision type",
                key="reg",
            )
            incident_severity = st.selectbox(
                "Select Incident severity:", severity_opt, help="Select severity type"
            )
            auto_make = st.selectbox(
                "Select auto make:", make_opt, help="Select Vehicle make"
            )
            occupation = st.selectbox(
                "Select occupation:", occu_opt, help="Customer occupation"
            )
            vehi_claim_amount = st.number_input(
                "Enter vehicle claim amount:",
                help="Enter vehicle claim amount",
                step=100,
            )
            inc_date = st.date_input("Select Incident  date", date.today())

        with col2:
            policy_premium = st.number_input(
                "Enter annual premium amount:",
                help="Enter annual premium amount",
                step=100,
            )
            cust_month = st.number_input(
                "Enter the Customer tenure ranges:",
                help="Enter the Customer tenure ranges. If new customer enter 0:",
                step=1,
            )
            cust_age = st.number_input(
                "Enter the Customer age:", help="Enter the Customer age:", step=1
            )
            hobbies = st.selectbox(
                "Select hobbies:", hobbies_opt, help="Select hobbies"
            )
            selected_date = st.date_input(
                "Select Insurance bind date", date.today(), min_value=date(1995, 1, 1)
            )

        severity = {
            "Trivial Damage": 1,
            "Major Damage": 2,
            "Minor Damage": 3,
            "Total Loss": 4,
        }

        incident_severity_map = severity[incident_severity]
        collision_type_encoded = encode_feature(
            collision_type, coll_type_en, "collision_type"
        )
        encoded_months = encode_months_as_customer(cust_month)
        auto_make_encoded = encode_feature(auto_make, auto_make_en, "auto_make")
        hobbies_encoded = encode_feature(hobbies, hobbies_en, "insured_hobbies")
        occupation_encoded = encode_feature(occupation, occu_en, "insured_occupation")
        insurance_age = inc_date.year - selected_date.year

        age_box = (
            transform_single_value(cust_age, lambda_dict.get("age_boxcox"))
            if cust_age and cust_age > 0
            else None
        )
        policy_premium_box = (
            transform_single_value(
                policy_premium, lambda_dict.get("policy_annual_premium_boxcox")
            )
            if policy_premium and policy_premium > 0
            else None
        )
        vehicle_claim_box = (
            transform_single_value(
                vehi_claim_amount, lambda_dict.get("vehicle_claim_boxcox")
            )
            if vehi_claim_amount and vehi_claim_amount > 0
            else None
        )

        st.markdown("<hr style='border: 4px solid #ffffff;'>", unsafe_allow_html=True)

        data_reg = np.array(
            [
                [
                    incident_severity_map,
                    collision_type_encoded,
                    policy_premium_box,
                    encoded_months,
                    age_box,
                    insurance_age,
                    auto_make_encoded,
                    hobbies_encoded,
                    occupation_encoded,
                    vehicle_claim_box,
                ]
            ]
        )
        # st.write('Regression Data:','\n',data_reg)

        button = (
            st.button("Get Predictions!")
            if (
                cust_age is not None
                and cust_month is not None
                and vehicle_claim_box is not None
                and policy_premium_box is not None
                and cust_age > 0
                and cust_month > 0
                and vehicle_claim_box > 0
                and policy_premium_box > 0
            )
            else st.markdown(texts, unsafe_allow_html=True)
        )
        preds = 0

        if (
            button
            and cust_age is not None
            and cust_month is not None
            and vehicle_claim_box is not None
            and policy_premium_box is not None
            and cust_age > 0
            and cust_month > 0
            and vehicle_claim_box > 0
            and policy_premium_box > 0
        ):
            preds = Reg_model.predict(data_reg)

            lambda_val = lambda_dict.get("total_claim_amount_boxcox")

            st.balloons()

            reversed_preds = reverse_boxcox_transform(preds, lambda_val)

            st.write(" ")

            st.markdown(
                f"""<p style='text-align: left; font-size: 22px; color: #ffffff; font-weight: 400; font-family: JetBrainsMono Nerd Font;text-indent: 0em;'>
                Predicted Claim amount is {reversed_preds:,.2f} USD </p>""",
                unsafe_allow_html=True,
            )

            st.snow()

    elif selected2 == "Fraud Detection":
        col1, cola, col2, colb, col3 = st.columns([2, 0.25, 2, 0.25, 2])

        with col1:
            cust_age = st.number_input(
                "Enter the Customer age:", help="Enter the Customer age:", step=1
            )
            cust_month = st.number_input(
                "Enter the Customer tenure ranges:",
                help="Enter the Customer tenure ranges. If new customer enter 0:",
                step=1,
            )
            insured_sex = st.selectbox(
                "Select gender:", ["Male", "Female"], help="Customer Gender"
            )
            education = st.selectbox(
                "Select education:", edu_opt, help="Customer education Level"
            )
            occupation = st.selectbox(
                "Select occupation:", occu_opt, help="Customer occupation"
            )
            hobbies = st.selectbox(
                "Select hobbies:", hobbies_opt, help="Select hobbies"
            )
            insured = st.selectbox(
                "Select insured relation:", insured_opt, help="Select insured relation"
            )
            auto_make = st.selectbox(
                "Select auto make:", make_opt, help="Select Vehicle make"
            )
            year = st.selectbox(
                "Select make year:",
                [i for i in range(1994, 2016)],
                help="Select Vehicle make year",
            )

        with col2:
            policy_premium = st.number_input(
                "Enter annual premium amount:",
                help="Enter annual premium amount",
                step=100,
            )
            selected_date = st.date_input(
                "Select Insurance bind date", date.today(), min_value=date(1995, 1, 1)
            )
            policy_deduc = st.selectbox(
                "Select deductable:",
                policy_deduc_opt,
                help="Portion of a claim that policy holder responsible to pay.",
            )
            policy_state = st.selectbox(
                "Select policy State:", states, help="Select Policy State/Location"
            )
            inc_date = st.date_input("Select Incident  date", date.today())
            inc_State = st.selectbox(
                "Incident State:",
                inc_state_opt,
                help="Select the state where the incident occured.",
            )
            city = st.selectbox(
                "Incident City:", city_opt, help="City where the incodent occured."
            )
            incident_type = st.selectbox(
                "Select incident type:", incident_opt, help="Select incident type"
            )
            incident_severity = st.selectbox(
                "Select Incident severity:", severity_opt, help="Select severity type"
            )

        with col3:
            hour = st.selectbox(
                "Incident Time:", hour_opt, help="Time when the incodent occured."
            )
            no_of_veh = st.selectbox(
                "No of Vehicle Involved:",
                vehicle_opt,
                help="Vehicles count that met with an incident.",
            )
            prpty_dmg = st.selectbox(
                "Property Damage:",
                prpty_dmg_opt,
                help="Any property damaged due to the incident.",
            )
            injury = st.selectbox("Injury:", injury_opt, help="No of people injured.")
            wit = st.selectbox(
                "No of witness:", wit_opt, help="No of witness for the incident."
            )
            fir = st.selectbox("Police Report:", fir_opt, help="Reported to police.")
            collision_type = st.selectbox(
                "Select Collision type:", collision_opt, help="Select Collision type"
            )
            auth = st.selectbox(
                "Authority Contacted:",
                auth_opt,
                help="Has any goverment authority contacted?",
            )
            st.write(" ")
            class_button = (
                st.button("Detector")
                if (
                    policy_premium is not None
                    and cust_age is not None
                    and cust_month is not None
                    and policy_premium > 0
                    and cust_age > 0
                    and cust_month > 0
                )
                else st.markdown(texts, unsafe_allow_html=True)
            )

        edu_en = edu_en_class.transform([education])
        severity_en = severity_en_class.transform([incident_severity])

        age_box = (
            transform_single_value(cust_age, lambda_dict.get("age_boxcox"))
            if cust_age and cust_age > 0
            else None
        )
        policy_premium_box = (
            transform_single_value(
                policy_premium, lambda_dict.get("policy_annual_premium_boxcox")
            )
            if policy_premium and policy_premium > 0
            else None
        )
        insurance_age = inc_date.year - selected_date.year
        vehi_age = inc_date.year - year
        sex = {"Male": 1, "Female": 0}
        property_damage_map = {"NO": 0, "YES": 1}
        police_report_available_map = {"NO": 0, "YES": 1}
        fraud_reported_map = {"N": 0, "Y": 1}

        policy_state_array = [0] * len(states)
        policy_state_selected_index = states.index(policy_state)
        policy_state_array[policy_state_selected_index] = 1

        coll_array = [0] * len(collision_opt)
        selected_index = collision_opt.index(collision_type)
        coll_array[selected_index] = 1

        inc_array = [0] * len(incident_opt)
        selected_inc_index = incident_opt.index(incident_type)
        inc_array[selected_inc_index] = 1

        rela_array = [0] * len(insured_opt)
        selected_rela_index = insured_opt.index(insured)
        rela_array[selected_rela_index] = 1

        hobbies_array = [0] * len(hobbies_opt)
        selected_hoobies_index = hobbies_opt.index(hobbies)
        hobbies_array[selected_hoobies_index] = 1

        occu_array = [0] * len(occu_opt)
        occ_selected_index = occu_opt.index(occupation)
        occu_array[occ_selected_index] = 1

        auth_contact_array = [0] * len(auth_opt)
        auth_contact_selected_index = auth_opt.index(auth)
        auth_contact_array[auth_contact_selected_index] = 1

        inc_state_array = [0] * len(inc_state_opt)
        inc_state_array_selected_index = inc_state_opt.index(inc_State)
        inc_state_array[inc_state_array_selected_index] = 1

        inc_city_array = [0] * len(city_opt)
        inc_city_selected_index = city_opt.index(city)
        inc_city_array[inc_city_selected_index] = 1

        auto_make_array = [0] * len(make_opt)
        auto_make_selected_index = make_opt.index(auto_make)
        auto_make_array[auto_make_selected_index] = 1

        data_class = np.array(
            [
                [
                    age_box,
                    policy_premium_box,
                    insurance_age,
                    vehi_age,
                    year,
                    policy_deduc,
                    cust_month,
                    sex[insured_sex],
                    edu_en[0],
                    severity_en[0],
                    hour,
                    no_of_veh,
                    property_damage_map[prpty_dmg],
                    injury,
                    wit,
                    police_report_available_map[fir],
                ]
                + policy_state_array
                + hobbies_array
                + occu_array
                + rela_array
                + inc_array
                + coll_array
                + auth_contact_array
                + inc_state_array
                + inc_city_array
                + auto_make_array
            ]
        )

        # Indices_mapper = ['age_box', 'policy_premium_box', 'insurance_age',' vehi_age', 'year', 'policy_deduc', cust_month, sex[insured_sex],
        #                        edu_en[0],severity_en[0],hour, no_of_veh, property_damage_map[prpty_dmg], injury, wit, police_report_available_map[fir],
        #                        ]
        # continuous = ['age_box', 'policy_premium_box', 'insurance_age', 'vehi_age', 'year', 'policy_deduc']

        # continuous_indices = [data_class.index(col) for col in continuous]

        dummy_features = [0] * 4

        scale_data = np.hstack(
            (data_class[:, :2], np.array([dummy_features]), data_class[:, 2:6])
        )

        # st.write('Classification Data:','\n',data_class)

        # st.write(None not in scale_data)
        if class_button and None not in scale_data:
            scaled_data_class = class_scale.transform(scale_data)

            # st.write('Classification Data Scaled:','\n',scaled_data_class)

            scaled_data_class_required = np.delete(
                scaled_data_class, [2, 3, 4, 5], axis=1
            )

            data_class_refined = np.delete(data_class, [0, 1, 2, 3, 4, 5], axis=1)

            final_data = np.hstack((scaled_data_class_required, data_class_refined))

            predict = Class_model.predict(final_data)

            # st.write(final_data,'\n', predict)

            if predict == 0:
                st.success(
                    "The transaction is predicted to be legitimate. No fraudulent activity detected."
                )

            else:
                st.warning(
                    "Alert! The transaction is predicted to be fraudulent. Please review it carefully."
                )

        else:
            pass
            # st.markdown(texts,unsafe_allow_html=True)
