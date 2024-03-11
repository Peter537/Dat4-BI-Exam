import streamlit as st
import datacleaner
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Data Processing",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Data Processing')

data_path = "./data/data_science_salaries.csv"
df, dfNumeric, dfNoOutliers = datacleaner.load_data(data_path)

min_work_year = df['work_year'].min()
max_work_year = df['work_year'].max()

st.write(f"The range of work years is from {min_work_year} to {max_work_year}.")

st.write("The following is a table of a sample of 5 rows in the dataset.")
st.write(df.sample(5))

st.write("We have two numeric colums in the dataset, namely 'salary_in_usd' and 'work_year', here we describe the statistics of these columns.")
st.write(df.describe())

fig = px.histogram(df['salary_in_usd'], x='salary_in_usd', nbins=50, title='Salary in USD')
st.plotly_chart(fig)

fig = px.histogram(dfNoOutliers, x='salary_in_usd', color='company_location',
                        title='Salary Distribution by Country',
                        labels={'salary_in_usd': 'Salary in USD', 'company_location': 'Country'},
                        template='plotly_white')
st.plotly_chart(fig)

st.write("We are in Denmark, so we will check all the danish salaries.")
danish_salaries = df[df['company_location'] == 'Denmark']
st.write(danish_salaries)
st.write(danish_salaries[['salary_in_usd']].describe())

fig = px.box(df, x='work_models', y='salary_in_usd', title='Salary Distribution by Work Model', labels={'work_models': 'Work Model', 'salary_in_usd': 'Salary (USD)'})
st.plotly_chart(fig)

st.write(df[['work_models', 'salary_in_usd']].groupby('work_models')['salary_in_usd'].describe())

# Define custom category order
category_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']

# Box Plot using Plotly Express
fig1 = px.box(df, x='experience_level', y='salary_in_usd', title='Salary Distribution by Experience Level', labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary'}, category_orders={'experience_level': category_order})
st.plotly_chart(fig1)

# Calculate average salary per job title
avg_salary_per_job = df.groupby('job_title')['salary_in_usd'].mean().reset_index()

# Bar Plot using Plotly Express
fig = px.bar(avg_salary_per_job, x='job_title', y='salary_in_usd', title='Average Salary by Job Title', labels={'job_title': 'Job Title', 'salary_in_usd': 'Average Salary'})
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig)

job_titles = df['job_title'].unique()

# Create a TfidfVectorizer to convert job titles into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(job_titles)

# Calculate distortions (inertia) for different values of k
distortions = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
