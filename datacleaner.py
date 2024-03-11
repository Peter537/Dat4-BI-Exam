import pandas as pd
import z_score

def return_cleaned_data(path):
    df = pd.read_csv(path)

    print("--- Check for missing values ---")
    print(df.isnull().sum())

    print("--- Check for NaN values ---")
    print(df.isna().sum())

    print("--- Check for data types ---")
    print(df.info())

    print("--- Check for shape ---")
    print(df.shape)

    # This column is not useful for our analysis and proves irrelevant
    df.drop(['employee_residence'], axis=1, inplace=True)

    # This column is redundant as we have another column explicitly in standardized USD
    df.drop(['salary', 'salary_currency'], axis=1, inplace=True)

    for col in df.columns:
        print("col at  ", col, " : ", df[col].dtype == "object")
        if df[col].dtype == 'object':
            df[col] = df[col].astype("string") # str virker ikke?

    print(df.info())
    # Create a Numeric DataFrame
    dfNumeric = pd.get_dummies(df, columns=['job_title', 'experience_level', 'employment_type', 'work_models', 'company_location', 'company_size'], dtype=pd.Int64Dtype())

    # Create a DataFrame without outliers
    dfNoOutliers = df.copy()
    dfNoOutliers['salary_in_usd'] = z_score.calculateList(df['salary_in_usd'], drop=True)

    return df, dfNumeric, dfNoOutliers

df, dfNumeric, dfNoOutliers = return_cleaned_data('data/data_science_salaries.csv')
