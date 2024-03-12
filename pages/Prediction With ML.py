import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pickle
import glob

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
dfClassification = st.session_state['dfNumeric'].copy()

try:
    st.warning("If this is the first time you are running this app, it may take a while to load the models as they will be trained. Please be patient.")

    if glob.glob("regression.pkl"):
        regression = pickle.load(open("regression.pkl", "rb"))
        pass
    else:
        # Add regression here
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
    st.write("Regression")

with tab2:
    st.title("Clustering")

    if st.button("Fix NS_ERROR_FAILURE or a white box below"): # This is a workaround for a bug in Streamlit. I can't figure out why the error appears, but reloading the data fixes it
        dfCluster = dfCluster.sample(frac=1).reset_index(drop=True)
        st.write("Data has been randomized")
    template = dfCluster.iloc[:1].copy()
    prediction = st.data_editor(template, key="dfCluster")

    X = dfCluster.copy()

    if (st.button("Predict cluster")):
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