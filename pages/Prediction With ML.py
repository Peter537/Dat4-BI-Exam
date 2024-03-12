import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pickle
import glob
import datacleaner 

st.set_page_config(
    page_title="Prediction With ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)





st.write("")
regression = None
classification = None
kmeans = None
dfCluster = st.session_state['dfNumeric'].drop(['gdp_per_capita'], axis=1)

df = datacleaner.load_data("./data/data_science_salaries.csv")

try:
    if glob.glob("regression.pkl"):
        regression = pickle.load(open("regression.pkl", "rb"))
        pass
    else:
        # Add regression here
        pass

    if glob.glob("cluster.pkl"):
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
    else:
        rowCluster = pd.read_csv("data/cluster.csv")
        df = st.session_state['dfNumeric'].copy()
        df['cluster'] = rowCluster['cluster']
        st.write(df)
        pass

except Exception as e:
    st.write("An error occurred while loading models: ", e)

tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Clustering", "Classification", "About"])

with tab1:
    df = st.session_state["df"]
    st.write("Regression")

with tab2:
    st.title("Clustering")

    if st.button("Fix NS_ERROR_FAILURE or a white box below"): # This is a workaround for a bug in Streamlit. I can't figure out why the error appears, but reloading the data fixes it
        dfCluster = dfCluster.sample(frac=1).reset_index(drop=True)
        st.write("Data has been randomized")
    template = dfCluster.iloc[:1].copy()


   
   # Dropdown for Job Title
    job_title_input = st.selectbox("Job Title", df['job_title'].unique())

    # Dropdown for Experience Level
    experience_level_input = st.selectbox("Experience Level", df['experience_level'].unique())

    # Dropdown for Company Location
    company_location_input = st.selectbox("Company Location", df['company_location'].unique())

    # Dropdown for Work Model
    work_model_input = st.selectbox("Work Model", df['work_models'].unique())
    # Dropdown for Work Year
    work_year_input = st.selectbox("Work Year", df['work_year'].unique())
   


    def createNewRow(job_title, experience_level, company_location, work_model, work_year):
        inputs = {}
        values = [job_title, experience_level, company_location, work_model, work_year]

        prefixes = ['job_title_', 'experience_level_', 'company_location_', 'work_models_', 'work_year_']
        
        # Iterate through prefixes and input values to create the input dictionary
        for prefix, value in zip(prefixes, values):
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

        return input_row


    X = dfCluster.copy()

    if (st.button("Predict cluster")):
        prediction = createNewRow(job_title_input, experience_level_input, company_location_input, work_model_input, work_year_input)
        st.write("The predicted cluster for the selected data is: " + kmeans.predict(prediction).__str__())

    st.title("KMeans Clustering Analysis")

    from yellowbrick.cluster import SilhouetteVisualizer
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X)
    fig = visualizer._fig
    st.pyplot(fig)
    st.write("The silhouette score of the model is: " + round(visualizer.silhouette_score_*100, 2).__str__() + "%")

with tab3:
    df = st.session_state["df"]
    st.write("Classification")

with tab4:
    st.write("About")
    

    

