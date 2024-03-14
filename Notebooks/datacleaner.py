import pandas as pd
import z_score

def load_data(path):
    df = pd.read_csv(path)

    # This column is not useful for our analysis and proves irrelevant
    df.drop(['employee_residence'], axis=1, inplace=True)

    # This column is redundant as we have another column explicitly in standardized USD
    df.drop(['salary', 'salary_currency'], axis=1, inplace=True)

    return df

def load_country_gdp_data(path):
    df = pd.read_csv(path)

    df = df[['Country Name', 'Year', 'GDP Per Capita']]
    df = df.rename(columns={'Country Name': 'country', 'Year': 'year', 'GDP Per Capita': 'gdp_per_capita'})

    # print cols where gdp is null
    df.dropna(subset=['gdp_per_capita'], inplace=True)

    df = df[df['year'] == 2022]

    df.drop(['year'], axis=1, inplace=True)

    return df

def combined_df():
    gdp_df = load_country_gdp_data("../data/country_gdp_data.csv")
    salary_df = load_data('../data/data_science_salaries.csv')

    notFoundList = []
    for country in salary_df['company_location'].unique():
        found = False
        for gdp_country in gdp_df['country'].unique():
            if country.lower() == gdp_country.lower():
                found = True
                break

        if not found:
            notFoundList.append(country)

    gdp_df.replace(to_replace="Russian Federation", value="Russia", inplace=True)
    gdp_df.replace(to_replace="Egypt, Arab Rep.", value="Egypt", inplace=True)
    gdp_df.replace(to_replace="Turkiye", value="Turkey", inplace=True)
    gdp_df.replace(to_replace="Korea, Rep.", value="South Korea", inplace=True)
    gdp_df.replace(to_replace="Hong Kong SAR, China", value="Hong Kong", inplace=True)
    gdp_df.replace(to_replace="Czechia", value="Czech Republic", inplace=True)
    gdp_df.replace(to_replace="Iran, Islamic Rep.", value="Iran", inplace=True)
    gdp_df.replace(to_replace="Bahamas, The", value="Bahamas", inplace=True)

    salary_df.drop(salary_df[salary_df['company_location'] == "Gibraltar"].index, inplace=True)

    notFoundList = []
    for country in salary_df['company_location'].unique():
        found = False
        for gdp_country in gdp_df['country'].unique():
            if country.lower() == gdp_country.lower():
                found = True
                break

        if not found:
            notFoundList.append(country)

    combined_df = pd.merge(salary_df, gdp_df, left_on='company_location', right_on='country', how='left')
    combined_df.drop(['country'], axis=1, inplace=True)

    for col in combined_df.columns:
        if combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].astype("string") # str virker ikke?

    df = get_no_outliers_df(combined_df)

    return df

def get_no_outliers_df(df):
    # Create a DataFrame without outliers
    dfNoOutliers = df.copy()
    dfNoOutliers['salary_in_usd'] = z_score.calculateList(df['salary_in_usd'], drop=True)
    return dfNoOutliers.dropna()

def get_numeric_df(df):
    # Create a Numeric DataFrame
    dfNumeric = pd.get_dummies(df, columns=['job_title', 'experience_level', 'employment_type', 'work_models', 'company_location', 'company_size'], dtype=pd.Int64Dtype())

    return dfNumeric

df = combined_df()