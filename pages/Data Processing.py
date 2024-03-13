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
    st.write("This shows us which columns in the dataset that are important.")
    st.write("We will start by looking at how many salaries there are in categories per 10k income.")
    fig = px.histogram(df['salary_in_usd'], x='salary_in_usd', nbins=50, title='Salary in USD')
    st.plotly_chart(fig)

with tab2:
    st.markdown("### Salary Count based on Country")
    fig = px.histogram(df, x='salary_in_usd', color='company_location',
                            title='Salary Distribution by Country',
                            labels={'salary_in_usd': 'Salary in USD', 'company_location': 'Country'},
                            template='plotly_white')
    st.plotly_chart(fig)

    average_salary_per_country = df.groupby('company_location')['salary_in_usd'].mean().reset_index()
    average_salary_per_country['medium_salary'] = df.groupby('company_location')['salary_in_usd'].median().values
    average_salary_per_country['location_count'] = df.groupby('company_location')['company_location'].count().values
    average_salary_per_country['gdp_per_capita'] = df.groupby('company_location')['gdp_per_capita'].mean().values
    average_salary_per_country = average_salary_per_country.sort_values(by='salary_in_usd', ascending=False)
    st.write(average_salary_per_country)

    # only the ones with average_salary_per_country['location_count'] > 10
    fig = px.scatter(average_salary_per_country[average_salary_per_country['location_count'] > 10], x='gdp_per_capita', y='salary_in_usd', size='location_count', color='company_location', text='company_location', title='Average Salary vs GDP per Capita by Country with over 10 datapoints')
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    st.markdown("#### Conclusion")
    st.write("From these graphs, we can see that America has the highest salaries, but there are also much more datapoints for America than any other country.")
    st.write("When we compare a Countries average salary to the GDP per capita, we can see that there is a correlation between the two.")
    st.write("Though Mexico, Russia and Colombia look to have a higher average salary than expected compared to GDP, but it can probably be explained because of the lact of data points.")

with tab3:
    st.markdown("### Salary Statistics by Work Model")
    st.write("")
    st.write(df[['work_models', 'salary_in_usd']].groupby('work_models')['salary_in_usd'].describe())
    fig = px.box(df, x='work_models', y='salary_in_usd', title='Salary Distribution by Work Model', labels={'work_models': 'Work Model', 'salary_in_usd': 'Salary (USD)'})
    st.plotly_chart(fig)

    st.markdown("#### Conclusion")
    st.write("We can see that on-site work has the highest average salary, which is expected, but it is interesting to see that the remote works salary distribution is so close to the on-site work.")
    st.write("It is also interesting to see that hydrid work has a much lower salary distribution than remote work, though it would be expected to be much lower than on-site work.")
    st.write("But even though there are many less datapoints for hydrid work, there are still enough to make a conclusion.")

with tab4:
    st.markdown("### Salary Distribution by Experience Level")
    # Define custom category order
    category_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']

    # Box Plot using Plotly Express
    fig1 = px.box(df, x='experience_level', y='salary_in_usd', title='Salary Distribution by Experience Level', labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary'}, category_orders={'experience_level': category_order})
    st.plotly_chart(fig1)

    st.markdown("#### Conclusion")
    st.write("This graph gives a good overview of how we would think the average salary would be for the different experience levels.")
    st.write("It is clear that the average salary increases with the experience level, which is something we expected comming into this analysis.")

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

    st.markdown("#### Conclusion")
    st.write("We can clearly see that the average salary varies a lot depending on the job title.")

with tab6:
    st.markdown("### GDP effect on Salary")
    # Create scatter plot using Plotly Express
    fig = px.scatter(df, x='gdp_per_capita', y='salary_in_usd', labels={'gdp_per_capita': 'GDP', 'salary_in_usd': 'Salary'})

    # Streamlit app
    st.title('Scatter Plot of GDP vs Salary')
    st.plotly_chart(fig)

    st.markdown("#### Conclusion")
    st.write("We can conclude that there is somewhat a correlation between GDP and Salary.")
    st.write("Usually, the countries with a lower GDP, more employees have a lower salary, whereas countries with a higher GDP than those, more employees have a higher salary.")
    st.write("Though we can see that in the countries with the highest GDP, they have a lower salary, but it might be due to the lack of data points or other factors.")

#Lav GDP = Større chance for du får en lav løn
#Højere GDP = Større chance for du får en højere løn, 
            # men stadig mange som får en mindre løn, mange faktorer spiller ind pga. forskelllige jobs, erfaring etc.
#Højest GDP = Vi har ikke nok data punkter til at kunne sige meget om det.
