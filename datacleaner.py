import pandas as pd
import z_score

def load_data(path):
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

    return df

def load_country_gdp_data(path):
    df = pd.read_csv(path)

    print("--- Check for data types ---")
    print(df.info())

    print("--- Check for shape ---")
    print(df.shape)

    df = df[['Country Name', 'Year', 'GDP Per Capita']]
    df = df.rename(columns={'Country Name': 'country', 'Year': 'year', 'GDP Per Capita': 'gdp_per_capita'})

    print("--- Check for missing values ---")
    print(df.isnull().sum())

    print("--- Check for NaN values ---")
    print(df.isna().sum())

    print("--- Check for data types ---")
    print(df.info())

    print("--- Check for shape ---")
    print(df.shape)

    # print cols where gdp is null
    df.dropna(subset=['gdp_per_capita'], inplace=True)

    print("--- Check for missing values ---")
    print(df.isnull().sum())

    print("--- Check for shape ---")
    print(df.shape)

    print(df['country'].unique())

    df = df[df['year'] == 2022]

    print(df['year'].unique())
    print(df.shape)

    df.drop(['year'], axis=1, inplace=True)

    return df

def combined_df():
    gdp_df = load_country_gdp_data("data/country_gdp_data.csv")
    salary_df = load_data('data/data_science_salaries.csv')

    notFoundList = []
    for country in salary_df['company_location'].unique():
        found = False
        for gdp_country in gdp_df['country'].unique():
            if country.lower() == gdp_country.lower():
                found = True
                break

        if not found:
            notFoundList.append(country)

    print("Countries not found in gdp data: ", notFoundList)
    gdp_df.replace(to_replace="Russian Federation", value="Russia", inplace=True)
    gdp_df.replace(to_replace="Egypt, Arab Rep.", value="Egypt", inplace=True)
    gdp_df.replace(to_replace="Turkiye", value="Turkey", inplace=True)
    gdp_df.replace(to_replace="Korea, Rep.", value="South Korea", inplace=True)
    gdp_df.replace(to_replace="Hong Kong SAR, China", value="Hong Kong", inplace=True)
    gdp_df.replace(to_replace="Czechia", value="Czech Republic", inplace=True)
    gdp_df.replace(to_replace="Iran, Islamic Rep.", value="Iran", inplace=True)
    gdp_df.replace(to_replace="Bahamas, The", value="Bahamas", inplace=True)

    print(salary_df[salary_df['company_location'] == "Gibraltar"])
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

    print("Countries not found in gdp data: ", notFoundList)

    combined_df = pd.merge(salary_df, gdp_df, left_on='company_location', right_on='country', how='left')
    combined_df.drop(['country'], axis=1, inplace=True)

    print(combined_df.info())
    print(combined_df.shape)
    print(combined_df.sample(5))

    df = get_no_outliers_df(combined_df)

    print(df.info())

    df = df.dropna()

    print(df.info())

    return df

def get_numeric_df(df):
    # Create a Numeric DataFrame
    dfNumeric = pd.get_dummies(df, columns=['job_title', 'experience_level', 'employment_type', 'work_models', 'company_location', 'company_size'], dtype=pd.Int64Dtype())

    return dfNumeric

def get_no_outliers_df(df):
    # Create a DataFrame without outliers
    dfNoOutliers = df.copy()
    dfNoOutliers['salary_in_usd'] = z_score.calculateList(df['salary_in_usd'], drop=True)

    return dfNoOutliers

#df = load_data('data/data_science_salaries.csv')

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

#dfGDP = load_country_gdp_data("data/country_gdp_data.csv")
#print(df["company_location"].unique())

df = combined_df()