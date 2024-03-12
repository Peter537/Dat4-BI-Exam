import streamlit as st
from sklearn.cluster import KMeans
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

try:
    if glob.glob("cluster.pkl"):
        kmeans = pickle.load(open("cluster.pkl", "rb"))
    else:
        kmeansX = dfCluster.copy()

        num_clusters = 9 # Higher number of clusters for more detailed analysis. Lower number of clusters for more general analysis. Should be 3 for more general analysis.
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
        kmeans.fit(kmeansX)

        pickle.dump(kmeans, open("cluster.pkl", "wb"))

except Exception as e:
    st.write("An error occurred: ", e)

tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Classification", "Clustering", "About"])

with tab1:
    df = st.session_state["df"]
    st.write("Regression")

with tab2:
    df = st.session_state["df"]
    st.write("Classification")

with tab3:
    st.title("Clustering")

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

with tab4:
    st.write("About")