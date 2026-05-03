import joblib
import pandas as pd
import streamlit as st


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="USA Rental Price Analyzer",
    page_icon="🏠",
    layout="wide"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = joblib.load("models/rent_model.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, columns


model, columns = load_model()


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .main-title {
        font-size: 46px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #475569;
        margin-bottom: 30px;
        max-width: 900px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 35px;
        border-radius: 24px;
        color: white;
        box-shadow: 0px 12px 30px rgba(15, 23, 42, 0.25);
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-text {
        font-size: 17px;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .info-card {
        background-color: white;
        padding: 26px;
        border-radius: 22px;
        box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 20px;
        height: 100%;
    }

    .info-card h3 {
        color: #0f172a;
        font-size: 23px;
        margin-bottom: 12px;
    }

    .info-card p {
        color: #475569;
        font-size: 16px;
        line-height: 1.6;
    }

    .result-card {
        background-color: white;
        padding: 32px;
        border-radius: 24px;
        box-shadow: 0px 10px 28px rgba(15, 23, 42, 0.12);
        text-align: center;
        margin-top: 22px;
        border: 1px solid #e2e8f0;
    }

    .rent-price {
        font-size: 46px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .rent-label {
        font-size: 17px;
        color: #64748b;
        font-weight: 600;
    }

    .category-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 12px 18px;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
        margin-top: 15px;
    }

    .category-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 12px 18px;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
        margin-top: 15px;
    }

    .category-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 12px 18px;
        border-radius: 999px;
        font-weight: 800;
        display: inline-block;
        margin-top: 15px;
    }

    .metric-box {
        background-color: white;
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
    }

    .metric-number {
        font-size: 32px;
        font-weight: 900;
        color: #0f172a;
    }

    .metric-label {
        color: #64748b;
        font-size: 15px;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPER FUNCTIONS
# =========================

def get_rent_category(rent):
    if rent < 1000:
        return "Low Rent", "category-low"
    elif rent < 2500:
        return "Medium Rent", "category-medium"
    else:
        return "High Rent", "category-high"


def prepare_input(data):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    final_df = pd.DataFrame(columns=columns)
    final_df.loc[0] = 0

    for col in input_df.columns:
        if col in final_df.columns:
            final_df[col] = input_df[col]

    return final_df


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🏠 Rent Analyzer")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Rent",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("USA Rental Price Prediction")


# =========================
# HOME PAGE
# =========================

if page == "Home":
    st.markdown('<div class="main-title">USA Rental Price Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">A machine learning web application that predicts estimated monthly rent in USD based on property features.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Predict Monthly Rental Prices in the USA</div>
            <div class="hero-text">
                This system estimates rental prices using property information such as location, property type,
                square feet, number of bedrooms, bathrooms, furnishing status, pet policy, laundry options,
                parking options, and accessibility features.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-number">$</div>
                <div class="metric-label">USD Rent Prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-number">RF</div>
                <div class="metric-label">Random Forest Model</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-number">ML</div>
                <div class="metric-label">Regression Project</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>What This App Does</h3>
                <p>The app allows users to enter property details and receive an estimated monthly rental price in dollars.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>How It Works</h3>
                <p>The system uses a Random Forest Regressor trained on USA rental housing data to estimate monthly rent.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# PREDICT RENT PAGE
# =========================

elif page == "Predict Rent":
    st.markdown('<div class="main-title">Predict Rent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Enter property details below to estimate the monthly rental price in USD.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        region = st.selectbox(
            "Region",
            [
                "new york city",
                "los angeles",
                "chicago",
                "houston",
                "phoenix",
                "san diego",
                "dallas",
                "austin",
                "seattle",
                "denver"
            ]
        )

        state = st.selectbox(
            "State",
            ["ny", "ca", "il", "tx", "az", "wa", "co", "fl", "ga", "nc"]
        )

        property_type = st.selectbox(
            "Property Type",
            ["apartment", "house", "condo", "townhouse", "duplex"]
        )

    with col2:
        sqfeet = st.number_input("Square Feet", min_value=100, max_value=10000, value=900)
        beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        baths = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

    with col3:
        cats_allowed = st.selectbox("Cats Allowed", ["No", "Yes"])
        dogs_allowed = st.selectbox("Dogs Allowed", ["No", "Yes"])
        smoking_allowed = st.selectbox("Smoking Allowed", ["No", "Yes"])
        wheelchair_access = st.selectbox("Wheelchair Access", ["No", "Yes"])
        electric_vehicle_charge = st.selectbox("EV Charging", ["No", "Yes"])
        comes_furnished = st.selectbox("Comes Furnished", ["No", "Yes"])

    col4, col5 = st.columns(2)

    with col4:
        laundry_options = st.selectbox(
            "Laundry Options",
            [
                "w/d in unit",
                "w/d hookups",
                "laundry in bldg",
                "laundry on site",
                "no laundry on site"
            ]
        )

    with col5:
        parking_options = st.selectbox(
            "Parking Options",
            [
                "off-street parking",
                "attached garage",
                "detached garage",
                "carport",
                "street parking",
                "no parking"
            ]
        )

    st.markdown("### Location Coordinates")

    col6, col7 = st.columns(2)

    with col6:
        lat = st.number_input("Latitude", min_value=20.0, max_value=50.0, value=40.7128)

    with col7:
        long = st.number_input("Longitude", min_value=-130.0, max_value=-60.0, value=-74.0060)

    if st.button("Predict Rental Price", use_container_width=True):
        input_data = {
            "region": region,
            "state": state,
            "type": property_type,
            "sqfeet": sqfeet,
            "beds": beds,
            "baths": baths,
            "cats_allowed": 1 if cats_allowed == "Yes" else 0,
            "dogs_allowed": 1 if dogs_allowed == "Yes" else 0,
            "smoking_allowed": 1 if smoking_allowed == "Yes" else 0,
            "wheelchair_access": 1 if wheelchair_access == "Yes" else 0,
            "electric_vehicle_charge": 1 if electric_vehicle_charge == "Yes" else 0,
            "comes_furnished": 1 if comes_furnished == "Yes" else 0,
            "laundry_options": laundry_options,
            "parking_options": parking_options,
            "lat": lat,
            "long": long
        }

        final_input = prepare_input(input_data)
        predicted_rent = model.predict(final_input)[0]
        predicted_rent = max(0, round(predicted_rent, 2))

        category, css_class = get_rent_category(predicted_rent)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="rent-label">Estimated Monthly Rent</div>
                <div class="rent-price">$ {predicted_rent:,.2f}</div>
                <div class="{css_class}">{category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This is an estimated rental price based on the dataset. It is for educational and project demonstration purposes only."
        )


# =========================
# ABOUT PAGE
# =========================

elif page == "About Project":
    st.markdown('<div class="main-title">About Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-card">
            <h3>USA Rental Price Analyzer</h3>
            <p>This project is a machine learning web application that predicts monthly rental prices in USD using USA housing rental data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Technologies Used</h3>
                <p>Python, Streamlit, Pandas, Scikit-learn, Random Forest Regressor, and Joblib.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Dataset Features</h3>
                <p>The model uses features such as region, state, property type, square feet, bedrooms, bathrooms,
                pet policy, smoking permission, furnishing status, laundry, parking, and location coordinates.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="info-card">
            <h3>Project Purpose</h3>
            <p>The purpose of this project is to demonstrate a practical regression machine learning application
            for rental price analysis. It includes data cleaning, encoding categorical features, model training,
            prediction, and web application development.</p>
        </div>
        """,
        unsafe_allow_html=True
    )