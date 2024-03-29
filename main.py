import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_option_menu import option_menu

import glob

st.set_page_config(
    page_title="BI Exam Project",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Choose a page!", divider='rainbow')

banner = """
    <body style="background-color:yellow;">
            <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); ;padding:10px">
                <h2 style="color:white;text-align:center;"> BI Exam Project </h2>
                <h3 style="color:white;text-align:center;"> By Magnus, Peter and Yusuf </h3>
                <div style="text-align:center">
                    <span style="color:white;text-align:center;"> This project contains algorithms for exploring data on salaries and relations with country data. </span>
                </div>
            </div>
    </body>
    <br>
    """

st.markdown(banner, unsafe_allow_html=True)

try:
    from datacleaner import load_data, load_country_gdp_data, combined_df, get_numeric_df

    if glob.glob("data/data_science_salaries.*"):
        st.session_state['df'] = load_data('data/data_science_salaries.csv')
    else:
        raise FileNotFoundError("File not found")
    if glob.glob("data/country_gdp_data.*"):
        st.session_state['dfGDP'] = load_country_gdp_data("data/country_gdp_data.csv")
    else:
        raise FileNotFoundError("File not found")
    st.session_state['dfCombined'] = combined_df()
    st.session_state['dfNumeric'] = get_numeric_df(st.session_state['dfCombined'])
except:
    st.error("Error loading data. Please make sure the data files are in the data folder and try again. \n\n The data folder should contain: \n\n data_science_salaries.csv \n\n country_gdp_data.csv")
    st.error("If these errors persist, feel free to contact the developers at cph-mk797@cphbusiness.dk")