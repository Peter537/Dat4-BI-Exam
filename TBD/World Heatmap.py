import streamlit as st
import mismatchFinder as mf
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth
import datacleaner
from streamlit_folium import st_folium

# Create a Streamlit app
st.set_page_config(
    page_title="World Heatmap",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

data_path = "./data/data_science_salaries.csv"
df, dfNumeric, dfNoOutliers = datacleaner.load_data(data_path)


# Load your DataFrame
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Your existing code
worldNames = world['name'].unique()
threshhold = 81
df = mf.find_and_correct_mismatches(worldNames, df, threshhold)

# Group by country and calculate the average salary
avg_salary_by_country = df.groupby('company_location')['salary_in_usd'].mean().reset_index()

# Merge GeoDataFrame with average salary information
merged_df = pd.merge(world, avg_salary_by_country, left_on='name', right_on='company_location', how='left')


st.title('World Heatmap')

choropleth_map = folium.Map(location=[0, 0], zoom_start=2)

Choropleth(
    geo_data=merged_df,
    name='choropleth',
    data=merged_df,
    columns=['company_location', 'salary_in_usd'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Salary in USD',
).add_to(choropleth_map)

folium.LayerControl().add_to(choropleth_map)

