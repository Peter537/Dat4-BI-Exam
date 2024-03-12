import streamlit as st
import datacleaner
import plotly.express as px

st.set_page_config(
    page_title="Data Processing",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Data Processing')

df = st.session_state["dfCombined"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Viewing Data", "Salary Count based on Country", "Salary Statistics by Work Model", "Salary Distribution by Experience Level", "Salary Statistics by Job Title", "GDP effect on Salary"])

with tab1:
    st.write("We will start by seeing what data we are working with, so here we look at the 5 first rows.")
    st.write(df.head())
    st.write("This shows us which columns in the dataset that are import ")
    st.write("We will start by looking at how many salaries there are in categories per 10k income.")
    fig = px.histogram(df['salary_in_usd'], x='salary_in_usd', nbins=50, title='Salary in USD')
    st.plotly_chart(fig)

    #st.write("We have two numeric colums in the dataset, namely 'salary_in_usd' and 'work_year', here we describe the statistics of these columns.")
    #st.write(df.describe())

with tab2:
    st.markdown("### Salary Count based on Country")
    st.write("Now we want to see how it is compared to the countries.")
    fig = px.histogram(df, x='salary_in_usd', color='company_location',
                            title='Salary Distribution by Country',
                            labels={'salary_in_usd': 'Salary in USD', 'company_location': 'Country'},
                            template='plotly_white')
    st.plotly_chart(fig)
    st.write("Here we see that America has most of the values, but we can also see there are fewer American values compared to other countries in the lower end, while there are more in the middle and upper end.")

    average_salary_per_country = df.groupby('company_location')['salary_in_usd'].mean().reset_index()
    average_salary_per_country['medium_salary'] = df.groupby('company_location')['salary_in_usd'].median().values
    average_salary_per_country['location_count'] = df.groupby('company_location')['company_location'].count().values
    average_salary_per_country['gdp_per_capita'] = df.groupby('company_location')['gdp_per_capita'].mean().values
    average_salary_per_country = average_salary_per_country.sort_values(by='salary_in_usd', ascending=False)
    st.write(average_salary_per_country)
    st.write("This gives a good overview because there are many countries which has very few salaries.")

    # only the ones with average_salary_per_country['location_count'] > 10
    st.write("Since some countries have very few salaries, we will only look at the ones with more than 10 salaries.")
    fig = px.scatter(average_salary_per_country[average_salary_per_country['location_count'] > 10], x='gdp_per_capita', y='salary_in_usd', size='location_count', color='company_location', text='company_location', title='Average Salary vs GDP per Capita by Country')
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.write("Looking at the graph, it makes sense that the countries with the highest GDP per capita would also have a higher salary")
    st.write("Mexico, Russia and Colombia look to be much higher than expected compared to GDP, but it can probably be explained because of the lact of data points.")

with tab3:
    st.markdown("### Salary Statistics by Work Model")
    st.write("")
    st.write(df[['work_models', 'salary_in_usd']].groupby('work_models')['salary_in_usd'].describe())
    fig = px.box(df, x='work_models', y='salary_in_usd', title='Salary Distribution by Work Model', labels={'work_models': 'Work Model', 'salary_in_usd': 'Salary (USD)'})
    st.plotly_chart(fig)

with tab4:
    st.markdown("### Salary Distribution by Experience Level")
    # Define custom category order
    category_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']

    # Box Plot using Plotly Express
    fig1 = px.box(df, x='experience_level', y='salary_in_usd', title='Salary Distribution by Experience Level', labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary'}, category_orders={'experience_level': category_order})
    st.plotly_chart(fig1)

with tab5:
#    st.markdown("### Salary Statistics by Job Title")
    # Calculate average salary per job title
#    avg_salary_per_job = df.groupby('job_title')['salary_in_usd'].mean().reset_index()

    # Bar Plot using Plotly Express
#    fig = px.bar(avg_salary_per_job, x='job_title', y='salary_in_usd', title='Average Salary by Job Title', labels={'job_title': 'Job Title', 'salary_in_usd': 'Average Salary'})
#    fig.update_xaxes(tickangle=45)
#    st.plotly_chart(fig)

    st.markdown("### Salary Statistics by Job Title")

    # Calculate average salary per job title
    avg_salary_per_job = df.groupby('job_title')['salary_in_usd'].mean().reset_index()

    # Sort by average salary in ascending order
    avg_salary_per_job = avg_salary_per_job.sort_values(by='salary_in_usd', ascending=True)

    # Bar Plot using Plotly Express
    fig = px.bar(avg_salary_per_job, x='job_title', y='salary_in_usd', title='Average Salary by Job Title', labels={'job_title': 'Job Title', 'salary_in_usd': 'Average Salary'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

with tab6:
    st.markdown("### GDP effect on Salary")
    # Create scatter plot using Plotly Express
    fig = px.scatter(df, x='gdp_per_capita', y='salary_in_usd', labels={'gdp_per_capita': 'GDP', 'salary_in_usd': 'Salary'})

    # Streamlit app
    st.title('Scatter Plot of GDP vs Salary')
    st.plotly_chart(fig)
