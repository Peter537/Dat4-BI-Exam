import streamlit as st

st.set_page_config(
    page_title="Prediction With ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)



tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Classification", "Clustering", "About"])

with tab1:
    st.write("Regression")

with tab2:
    st.write("Classification")

with tab3:
    st.write("Clustering")

with tab4:
    st.write("About")