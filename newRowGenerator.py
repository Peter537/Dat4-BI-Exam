import pandas as pd
import streamlit as st

def create_input_row(job_title_input, experience_level_input, company_location_input, df_columns):
    
    inputs = {}

    # Some prefixes so we can use the numeric df and input our own job titles in text
    prefixes = ['job_title_', 'experience_level_', 'company_location_']
    values = [job_title_input, experience_level_input, company_location_input]

    # Iterate through prefixes and input values to create the input dictionary
    for prefix, value in zip(prefixes, values):
        column_name = f"{prefix}{value}"
        if column_name in df_columns.columns:
            inputs[column_name] = 1
            # Set all other columns with the same prefix to 0
            for col in df_columns.columns:
                if col.startswith(prefix) and col != column_name:
                    inputs[col] = 0

    # Convert the dictionary to a Series and then to a DataFrame
    input_row = pd.Series(inputs)
    input_row = pd.DataFrame([input_row])

    # Ensure the input_row has the same columns as df, excluding the target variable
    input_row = input_row.reindex(columns=df_columns.columns.drop('salary_in_usd'), fill_value=0)

    return input_row



def createNewRow(job_title, experience_level, company_location, work_model, work_year, salaryUSD, employment_type, company_size, dfCluster):

    inputs = {}

    # Directly assign values for columns without prefixes
    direct_columns = ['work_year', 'salary_in_usd']
    direct_values = [work_year, salaryUSD]

    for direct_column, direct_value in zip(direct_columns, direct_values):
        inputs[direct_column] = direct_value

    prefixes = ['job_title_', 'experience_level_', 'company_location_', 'work_models_', 'employment_type_', 'company_size_']

    # Iterate through prefixes and input values to create the input dictionary
    for prefix, value in zip(prefixes, [job_title, experience_level, company_location, work_model, employment_type, company_size]):
        column_name = f"{prefix}{value}"
        if column_name in dfCluster.columns:
            inputs[column_name] = 1
            # Set all other columns with the same prefix to 0
            for col in dfCluster.columns:
                if col.startswith(prefix) and col != column_name:
                    inputs[col] = 0

    # Convert the dictionary to a Series and then to a DataFrame
    input_row = pd.Series(inputs)
    input_row = pd.DataFrame([input_row])

    # Ensure the input_row has the same columns as dfCluster
    input_row = input_row.reindex(columns=dfCluster.columns, fill_value=0)

    st.write(input_row)

    return input_row


def createNewClassRow(job_title2, experience_level2, company_location2, work_model2, work_year2, employment_type2, company_size2, cluster_input2, dfCombined, dfClassification):

    inputs = {}

    gdp = dfCombined[dfCombined['company_location'] ==  company_location2]['gdp_per_capita'][0]

    # Directly assign values for columns without prefixes
    direct_columns = ['work_year', 'cluster', 'gdp_per_capita']
    direct_values = [work_year2, cluster_input2, gdp]

    for direct_column, direct_value in zip(direct_columns, direct_values):
        inputs[direct_column] = direct_value

    prefixes = ['job_title_', 'experience_level_', 'company_location_', 'work_models_', 'employment_type_', 'company_size_']

    # Iterate through prefixes and input values to create the input dictionary
    for prefix, value in zip(prefixes, [job_title2, experience_level2, company_location2, work_model2, employment_type2, company_size2, cluster_input2, gdp]):
        column_name = f"{prefix}{value}"
        if column_name in dfClassification.columns:
            inputs[column_name] = 1
            # Set all other columns with the same prefix to 0
            for col in dfClassification.columns:
                if col.startswith(prefix) and col != column_name:
                    inputs[col] = 0

    # Convert the dictionary to a Series and then to a DataFrame
    input_row = pd.Series(inputs)
    input_row = pd.DataFrame([input_row])

    # Ensure the input_row has the same columns as dfCluster
    input_row = input_row.reindex(columns=dfClassification.columns, fill_value=0)

    st.write(input_row)

    return input_row