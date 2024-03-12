import streamlit as st
import datacleaner
import plotly.express as px

st.set_page_config(
    page_title="Data Exploration",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def charts():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Histogram', 'Boxplot', 'Scatterplot', 'Lineplot', 'Correlation Heatmap', "Barplot"])
    with tab1:
        fig = px.histogram(df, x=x, y=y, color=z, title='Histogram')
        st.plotly_chart(fig)
    with tab2:
        fig = px.box(df, x=x, y=y, color=z, title='Boxplot')
        st.plotly_chart(fig)
    with tab3:
        fig = px.scatter(df, x=x, y=y, color=z, title='Scatterplot')
        st.plotly_chart(fig)
    with tab4:
        fig = px.line(df, x=x, y=y, color=z, title='Lineplot')
        st.plotly_chart(fig)
    with tab5:
        fig = px.imshow(df[[x, y]].corr(), title='Correlation Heatmap')
        st.plotly_chart(fig)
    with tab6:
        fig = px.bar(df, x=x, y=y, color=z, title='Barplot')
        st.plotly_chart(fig)

def column_picker(df):
    st.header('Grouping by a Nominal Attribute')
    x = st.selectbox('**Select the nominal attribute, X**', df.columns)
    y = st.selectbox('**Select first measure, Y**', df.columns)
    z = st.selectbox('**Select extra measure, Z**', df.columns)     
    return x, y, z

df = st.session_state["dfCombined"]
st.title('Data Exploration')

x, y, z = column_picker(df)

if st.button(":green[Explore]"):
    st.subheader("Explore the Data in Diagrams")
    st.write('Click on tabs to explore')
    container = st.container()
    charts()