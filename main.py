import streamlit as st
from streamlit_option_menu import option_menu

import json
import requests
import pandas as pd
import numpy as np


from io import StringIO
import langdetect
from langdetect import DetectorFactory, detect, detect_langs
from PIL import Image

st.set_page_config(
    page_title="BI Exam Project",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#st.sidebar.header("Try Me!", divider='rainbow')
# st.sidebar.success("Select a demo case from above")
#st.image(logo, width=200)

banner = """
    <body style="background-color:yellow;">
            <div style="background-image: linear-gradient(90deg, rgb(255, 75, 75), rgb(28, 3, 204)); ;padding:10px">
                <h2 style="color:white;text-align:center;">BI Exam Project</h2>
                <h3 style="color:white;text-align:center;"> By Magnus, Peter and Yusuf </h3>
            </div>
    </body>
    <br>
    """

st.markdown(banner, unsafe_allow_html=True)

