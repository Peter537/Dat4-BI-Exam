import pandas as pd

def create_input_row(job_title_input, experience_level_input, company_location_input, df_columns):
    
    inputs = {}

    # Some prefixes so we can use the numeric df and input our own job titles in text
    prefixes = ['job_title_', 'experience_level_', 'company_location_']
    values = [job_title_input, experience_level_input, company_location_input]

    # Iterate through prefixes and input values to create the input dictionary
    for prefix, value in zip(prefixes, values):
        column_name = f"{prefix}{value}"
        if column_name in df_columns:
            inputs[column_name] = 1
            # Set all other columns with the same prefix to 0
            for col in df_columns:
                if col.startswith(prefix) and col != column_name:
                    inputs[col] = 0

    # Convert the dictionary to a Series and then to a DataFrame
    input_row = pd.Series(inputs)
    input_row = pd.DataFrame([input_row])

    # Ensure the input_row has the same columns as df, excluding the target variable
    input_row = input_row.reindex(columns=df_columns.drop('salary_in_usd'), fill_value=0)

    return input_row
