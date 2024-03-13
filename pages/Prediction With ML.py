import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, silhouette_score
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

        regression = RandomForestRegressor(n_estimators=50, random_state=116)
        regression.fit(X_train, y_train)
        pickle.dump(regression, open("regression.pkl", "wb"))
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

# ------------------- Regression -------------------

with tab1:
    df = st.session_state["df"]
    st.title("Random Forest Regressor")
    # Dropdown for Job Title
    job_title_input = st.selectbox("Job Titles", df['job_title'].unique() )
    # Dropdown for Experience Level
    experience_level_input = st.selectbox("Experience Levels", df['experience_level'].unique()) # Make the order alphabetical
    # Dropdown for Company Location
    company_location_input = st.selectbox("Company Locations", df['company_location'].unique())

    if (st.button("Predict Salary")):
        new_data_point = ng.create_input_row(job_title_input, experience_level_input, company_location_input, dfNum.columns)
        predicted_salary = regression.predict(new_data_point)[0]
        st.write(f"Predicted Salary: {predicted_salary:.2f} USD")

    st.title("Random Forest Regressor Analysis")
    st.write("We use a Regression model to predict the salary, because we need to predict the relationship between independent variables and dependent variables. The independent variables are the job title, experience level and company location, and the dependent variable is the salary in USD.")
    st.write("This will allow the user to input the job title, experience level and company location, and the model will predict what they could expect to earn in USD.")
    st.write("We use Random Forest Regressor instead of other Regressors, because it is a very powerful and accurate algorithm. It works by training many decision tree regressors, which means we will get a better prediction using this model than other comparable models.")
    st.write("\n")

    st.write("The Root Mean Squared Error (RMSE) gives the average salary deviation from the actual salary values, which is 48.344,79 USD.")
    st.write("Since our salary data is between 15.000 USD and 357.900 USD with a mean value of 142.835 USD, it isn't the best model, because it could be significantly off.")
    st.write("\n")

    st.write("R squared (R2) is the proportion of the variance in the dependent variable which is predictable from the independent variable. Since our R2 score is 41,28%, this means that our model can explain 41.28% of the variance in the dependent variable.")
    st.write("This means that our accuracy score i 41,28%, so in the case where it's wrong, it could be wrong withing our RMSE range.")

# ------------------- Clustering -------------------

with tab2:
    st.title("Clustering")

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

    st.write("K-Means was chosen because it is a simple and efficient algorithm. It is also easy to interpret the results. The algorithm works by dividing the data into clusters, where each cluster has its own centroid. Since this dataset is not very large, Hierarchical Clustering could also have been used, but being able to manually set the number of clusters is a big advantage of KMeans, and one that is needed for the purpose of this cluster, as explained later.")

    st.write("The final result was clustering the data into 9 clusters, which gave a silhouette score of 0.53, which is considered to be a good score.")
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
    krange = range(2, 12)
    with row[0]:
        scores = []
        for k in krange:
            model = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=42).fit(X)
            model.fit(X)
            score = silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
            scores.append(score)

        plot = plt.figure()
        plt.plot(krange, scores, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        st.pyplot(plot)

    with row[1]:
        distortions = []
        K = krange
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
    
    st.write("The amount of cluster numbers tested is limited to 11 as the significance of the clusters will be lost at a larger amount.")

    st.write("From the above graphs, we can see that the optimal number of clusters is 3. This is because the silhouette score is highest at 3 and the elbow graph shows an inflection point at 3.")
    st.write("Despite this, the number of clusters chosen was 9 to allow for more detailed analysis. The loss of choosing 9 instead of 3 is minor as the difference in silhouette score is less than 0.02.")
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
    st.title("About")
    
    st.write("Each tab contains a model that has already been trained and is ready to be used. Following the 'prediction' part, there is a section explaining what model is used and how accurate it is.")
    st.write("The data used for the models is a combination of two datasets: one containing salary data and the other containing information on countries. From the second dataset, the GDP per capita is used to create a new feature in the first dataset.")
    st.write("The data can be seen below, containg both the GDP and the result of clustering:")

    dfCombined = st.session_state['dfCombined']
    dfCombined['cluster'] = rowCluster['cluster']
    st.write(dfCombined)