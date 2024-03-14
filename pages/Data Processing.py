import streamlit as st
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
    st.write("Here you can see an overview of the cleaned data we are working with, it is a combination of the two datasets we have cleaned.")
    st.write(df)
    st.write("That is a good way to see the data, since we are focusing on salaries, this graph is a nice visualization of the salary distribution.")
    fig = px.histogram(df['salary_in_usd'], x='salary_in_usd', nbins=50, title='Salary in USD')
    st.plotly_chart(fig)

with tab2:
    st.markdown("### Salary Count based on Country")
    st.write("Press once on the country in the right to remove it.")
    st.write("Press twice on the country in the right to only show that country.")
    fig = px.histogram(df, x='salary_in_usd', color='company_location',
                            title='Salary Distribution by Country',
                            labels={'salary_in_usd': 'Salary (USD)', 'company_location': 'Country'},
                            template='plotly_white')
    st.plotly_chart(fig)

    country_count = df['company_location'].value_counts().reset_index()
    country_count.columns = ['Country', 'Count']
    country_count_sorted = country_count.sort_values(by='Count', ascending=False)
    total_entries = country_count_sorted['Count'].sum()
    country_count_sorted['Percentage'] = ((country_count_sorted['Count'] / total_entries) * 100).round(2).astype(str) + "%"
    st.write(country_count_sorted)

    st.markdown("#### Conclusion")
    st.write("This is a good indicator that most of our data points are from the United States. This means that maybe some of the data we have gotten from countries with few data points, they might not be representative.")

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
    category_order = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']

    fig1 = px.box(df, x='experience_level', y='salary_in_usd', title='Salary Distribution by Experience Level', labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary (USD)'}, category_orders={'experience_level': category_order})
    st.plotly_chart(fig1)

    st.markdown("#### Conclusion")
    st.write("This graph gives a good overview of how we would think the average salary would be for the different experience levels.")
    st.write("It is clear that the average salary increases with the experience level, which is something we expected coming into this analysis.")
    st.write("An interesting point is that the senior- and executive-level has the same low salary of 15.000, so maybe we should have looked at outliers in this case aswell, instead of only looking at outliers based on the salary.")

with tab5:
    st.markdown("### Salary Statistics by Job Title")

    avg_salary_per_job = df.groupby('job_title')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    avg_salary_per_job.columns = ['job_title', 'average_salary', 'count']
    avg_salary_per_job = avg_salary_per_job.sort_values(by='average_salary')

    fig = px.bar(avg_salary_per_job, x='job_title', y='average_salary', title='Average Salary by Job Title',
                    labels={'job_title': 'Job Title', 'average_salary': 'Average Salary (USD)'},
                    hover_data={'count': True, 'average_salary': ':.2f'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    st.write(avg_salary_per_job.sort_values(by='average_salary', ascending=False))

    avg_salary_per_job_filtered = avg_salary_per_job[avg_salary_per_job['count'] >= 10].sort_values(by='average_salary')

    fig_filtered = px.bar(avg_salary_per_job_filtered, x='job_title', y='average_salary', title='Average Salary by Job Title (Count >= 10)',
                        labels={'job_title': 'Job Title', 'average_salary': 'Average Salary (USD)'},
                        hover_data={'count': True, 'average_salary': ':.2f'})
    fig_filtered.update_xaxes(tickangle=45)
    st.plotly_chart(fig_filtered)

    st.markdown("#### Conclusion")
    st.write("We can clearly see that the average salary varies a lot depending on the job title.")
    st.write("It is interesting to see that the jobs with the lowest average salary has the fewest amount of data points, so it might not be as representative as the other job titles.")

with tab6:
    st.markdown("### GDP effect on Salary")
    average_salary_per_country = df.groupby('company_location')['salary_in_usd'].mean().reset_index()
    average_salary_per_country['medium_salary'] = df.groupby('company_location')['salary_in_usd'].median().values
    average_salary_per_country['location_count'] = df.groupby('company_location')['company_location'].count().values
    average_salary_per_country['gdp_per_capita'] = df.groupby('company_location')['gdp_per_capita'].mean().values

    filtered_data = average_salary_per_country[average_salary_per_country['location_count'] > 10]
    fig = px.scatter(filtered_data, x='gdp_per_capita', y='salary_in_usd', size='location_count', color='company_location', text='company_location', 
                 title='Average Salary vs GDP per Capita by Country with over 10 datapoints',
                 labels={'gdp_per_capita': 'GDP per Capita', 'salary_in_usd': 'Average Salary (USD)', 'location_count': 'Location Count'})
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    average_salary_per_country = average_salary_per_country.sort_values(by='salary_in_usd', ascending=False)
    st.write(average_salary_per_country)

    fig = px.scatter(df, x='gdp_per_capita', y='salary_in_usd', labels={'gdp_per_capita': 'GDP', 'salary_in_usd': 'Salary'})

    st.markdown("#### Conclusion")
    st.write("Looking at this graph, we can clearly see the correlation between the GDP per capita and the average salary in the traditional western countries.")
    st.write("But it is interesting to note that traditional less wealthy countries like Mexico, Russia, Colombia and Brazil have very high average salary. It might be because of the lack of datapoints from those countries, or it could be becuase the difference between the poor and the wealthy is very large, and these jobs are usually the ones with the higher salary, so the employees might be the ones with a higher social status.")
