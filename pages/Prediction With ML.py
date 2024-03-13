import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, silhouette_score
import pickle
import glob
from sklearn.ensemble import RandomForestRegressor
import newRowGenerator as ng
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediction With ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


df = st.session_state['df']


st.write("")
regression = None
classification = None
kmeans = None
dfCluster = st.session_state['dfNumeric'].drop(['gdp_per_capita'], axis=1)
dfClassification = st.session_state['dfNumeric'].copy()
dfNum = st.session_state['dfNumeric'].copy()

try:
    st.warning("If this is the first time you are running this app, it may take a while to load the models as they will be trained. Please be patient.")

    if glob.glob("regression.pkl"):
        regression = pickle.load(open("regression.pkl", "rb"))
        pass
    else:
        # Splitting the data into features (X) and target variable (y)
        X = dfNum.drop('salary_in_usd', axis=1)
        y = dfNum['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=83)

        rf_regressor = RandomForestRegressor(n_estimators=50, random_state=116)

        rf_regressor.fit(X_train, y_train)

        y_pred = rf_regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, y_pred)
        r2 = r2*100
        r2 = round(r2, 2)

        pass

    if glob.glob("cluster.pkl") and glob.glob("data/cluster.csv"):
        kmeans = pickle.load(open("cluster.pkl", "rb"))
    else:
        X = dfCluster.copy()

        num_clusters = 9 # Higher number of clusters for more detailed analysis. Lower number of clusters for more general analysis. Should be 3 for more general analysis.
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        y = kmeans.predict(X)
        rowCluster = pd.DataFrame(y, columns=['cluster'])

        rowCluster.to_csv("data/cluster.csv", index=False)
        pickle.dump(kmeans, open("cluster.pkl", "wb"))

    if glob.glob("classification.pkl"):
        classification = pickle.load(open("classification.pkl", "rb"))
        rowCluster = pd.read_csv("data/cluster.csv")
        dfClassification['cluster'] = rowCluster['cluster']
    else:
        rowCluster = pd.read_csv("data/cluster.csv")
        dfClassification['cluster'] = rowCluster['cluster']

        X = dfClassification.drop(['salary_in_usd'], axis=1)
        y = dfClassification['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

        classification = DecisionTreeClassifier(random_state=10)
        classification.fit(X_train, y_train)

        print("Accuracy: ", classification.score(X_test, y_test))
        print("RMSE:", root_mean_squared_error(y_test, classification.predict(X_test)))
        print("Classes: ", classification.classes_)

        pickle.dump(classification, open("classification.pkl", "wb"))

except Exception as e:
    st.write("An error occurred while loading models: ", e)

tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Clustering", "Classification", "About"])

with tab1:
    df = st.session_state["df"]
    st.title("Regression")
    #MSE telling us that a considerable amount of units discreptancy between predicted and actual values
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    #RMSE telling us that the salary deviates around 62900 USD from the actual values
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    #R2 is basically our accuracy score.
    st.write(f'R-squared (R2): {r2}')

    st.title("Regression Analysis")
    # Dropdown for Job Title
    job_title_input = st.selectbox("Job Titles", df['job_title'].unique() )
    # Dropdown for Experience Level
    experience_level_input = st.selectbox("Experience Levels", df['experience_level'].unique()) # Make the order alphabetical
    # Dropdown for Company Location
    company_location_input = st.selectbox("Company Locations", df['company_location'].unique())

    if (st.button("Predict Salary")):
#        new_data_point = ng.create_input_row('Marketing Data Scientist', 'Senior-level', 'United States', dfNum.columns)
        new_data_point = ng.create_input_row(job_title_input, experience_level_input, company_location_input, dfNum.columns)

        predicted_salary = rf_regressor.predict(new_data_point)[0]

        st.write(f'Predicted Salary: {predicted_salary} USD')

with tab2:
    st.title("Clustering")

    if st.button("Fix NS_ERROR_FAILURE or a white box below"): # This is a workaround for a bug in Streamlit. I can't figure out why the error appears, but reloading the data fixes it
        dfCluster = dfCluster.sample(frac=1).reset_index(drop=True)
        st.write("Page should be fixed now...")


   # Dropdown for Job Title
    job_title_input = st.selectbox("Job Title", df['job_title'].unique() )

    # Dropdown for Experience Level
    experience_level_input = st.selectbox("Experience Level", df['experience_level'].unique()) # Make the order alphabetical

    # Dropdown for salaryUSD
    salaryUSD_input = st.number_input("salaryUSD", min_value=1, max_value=500000, key='salary_in_usd')

    # Dropdown for employment type
    employment_type_input = st.selectbox("Employment Type", df['employment_type'].unique())

    # Dropdown for Work Model
    work_model_input = st.selectbox("Work Model", df['work_models'].unique())

    # Dropdown for Company Location
    company_location_input = st.selectbox("Company Location", df['company_location'].unique())

    # Dropdown for Work Year
    work_year_input = st.selectbox("Work Year", df['work_year'].unique())

    # Dropdown for Company Size
    company_size_input = st.selectbox("Company Size", df['company_size'].unique())

    def createNewRow(job_title, experience_level, company_location, work_model, work_year, salaryUSD, employment_type, company_size):

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



    X = dfCluster.copy()

    if (st.button("Predict cluster")):
        prediction = createNewRow(job_title_input, experience_level_input, company_location_input, work_model_input,
                                   work_year_input, salaryUSD_input, employment_type_input, company_size_input)
        
        st.write("The predicted cluster for the selected data is: " + kmeans.predict(prediction).__str__())
        # clusters = pd.DataFrame('Cluster': kmeans) something like this to combine the clusters with the original df.

    st.title("KMeans Clustering Analysis")

    st.write("For this analysis, we used KMeans clustering to group the data into 9 clusters. The silhouette score of the model is 0.53, which is considered to be a good score.")
    st.write("The silhouette score is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. \n\n If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters. In the case of a negative score, a point is placed in the wrong cluster compared to where it was expected")
    
    st.write("The silhouette score of the model can be visualised as follows:")

    from yellowbrick.cluster import SilhouetteVisualizer
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X)
    fig = visualizer._fig
    st.pyplot(fig)
    st.write("The silhouette score of the model is: " + round(visualizer.silhouette_score_*100, 2).__str__() + "%")

    st.write("For this cluster, a few possible sizes were considered as seen below in the silhouette score and elbow graph:")

    row = st.columns([1, 1])
    with row[0]:
        scores = []
        for k in range(2, 10):
            model = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=42).fit(X)
            model.fit(X)
            score = silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
            scores.append(score)

        plot = plt.figure()
        plt.plot(range(2,10), scores, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        st.pyplot(plot)

    with row[1]:
        distortions = []
        K = range(2,10)
        for k in K:
            model = KMeans(n_clusters=k, n_init=10).fit(X)
            model.fit(X)
            distortions.append(model.inertia_)
        
        plot2 = plt.figure()
        plt.title('Elbow Method for Optimal K')
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Distortion')
        st.pyplot(plot2)
    
    st.write("From the above graphs, we can see that the optimal number of clusters is 3. This is because the silhouette score is highest at 3 and the elbow graph shows an inflection point at 3.")
    st.write("Despite this, the number of clusters chosen was 9 to allow for more detailed analysis. The silhouette score of 0.53 is still considered to be a good score, and the lose is minimal.")
    st.write("Having a good describability of the clusters is very important, as the result is used for the classification model. The classification model is used to predict the salary of a new data point based on the cluster it belongs to. The more detailed the clusters are, the more accurate the classification model will be.")

# ------------------- Classification -------------------

with tab3:
    df['cluster'] = rowCluster['cluster']
    st.write("Classification")

    # Dropdown for Job Title
    job_title_input = st.selectbox("Job Title", df['job_title'].unique(), key='class_job_title')

    # Dropdown for Experience Level
    experience_level_input = st.selectbox("Experience Level", df['experience_level'].unique(), key='class_experience_level') # Make the order alphabetical

    # Dropdown for employment type
    employment_type_input = st.selectbox("Employment Type", df['employment_type'].unique(), key='class_employment_type')

    # Dropdown for Work Model
    work_model_input = st.selectbox("Work Model", df['work_models'].unique(), key='class_work_models')

    # Dropdown for Company Location
    company_location_input = st.selectbox("Company Location", df['company_location'].unique(), key='class_company_location')

    # Dropdown for Work Year
    work_year_input = st.selectbox("Work Year", df['work_year'].unique(), key='class_work_year')

    # Dropdown for Company Size
    company_size_input = st.selectbox("Company Size", df['company_size'].unique(), key='class_company_size')

    # Dropdown for cluster
    cluster_input = st.selectbox("Cluster", df['cluster'].unique(), key='class_cluster')


    def createNewRow(job_title, experience_level, company_location, work_model, work_year, employment_type, company_size):

        inputs = {}

        # Directly assign values for columns without prefixes
        direct_columns = ['work_year']
        direct_values = [work_year]

        for direct_column, direct_value in zip(direct_columns, direct_values):
            inputs[direct_column] = direct_value

        prefixes = ['job_title_', 'experience_level_', 'company_location_', 'work_models_', 'employment_type_', 'company_size_']

        # Iterate through prefixes and input values to create the input dictionary
        for prefix, value in zip(prefixes, [job_title, experience_level, company_location, work_model, employment_type, company_size]):
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

    if (st.button("Predict salary")):
        prediction = createNewRow(job_title_input, experience_level_input, company_location_input, work_model_input,
                                   work_year_input, employment_type_input, company_size_input)
        st.write("The predicted salary for the selected data is: " + classification.predict(prediction).__str__())

    st.title("Classification Analysis")

with tab4:
    st.write("About")
    

    

