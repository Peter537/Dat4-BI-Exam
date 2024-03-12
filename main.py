import streamlit as st
from streamlit_option_menu import option_menu

import datacleaner

import json
import requests
import pandas as pd
import numpy as np

from io import StringIO
import langdetect
from langdetect import DetectorFactory, detect, detect_langs
from PIL import Image
from datacleaner import load_data, load_country_gdp_data, combined_df

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

st.session_state['df'] = load_data('data/data_science_salaries.csv')
st.session_state['dfGDP'] = load_country_gdp_data("data/country_gdp_data.csv")
st.session_state['dfCombined'] = combined_df()