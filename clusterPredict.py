import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class JobClusterPredictor:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None
        self.clustered_titles = None

    def train_cluster_model(self, optimal_k=4):
        job_titles = self.df['job_title'].unique()

        # Convert job titles into numerical features using TfidfVectorizer
        X = self.vectorizer.fit_transform(job_titles)

        # Train KMeans clustering model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.kmeans.fit(X)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        # Create a dataframe with clustered job titles and PCA components
        self.clustered_titles = pd.DataFrame({'Job Title': job_titles, 'Cluster': self.kmeans.labels_})
        self.clustered_titles['PCA1'] = X_pca[:, 0]
        self.clustered_titles['PCA2'] = X_pca[:, 1]

    def predict_cluster(self, job_title, experience_level, company_location):
        # Transform input data to match the format used for training
        input_data = pd.DataFrame({'Job Title': [job_title], 'Cluster': [0]})
        input_data['Experience Level'] = experience_level
        input_data['Company Location'] = company_location

        # Use the trained model's vectorizer to transform job title
        input_title_vectorized = self.vectorizer.transform(input_data['Job Title'])

        # Apply PCA for dimensionality reduction
        input_title_pca = PCA(n_components=2).fit_transform(input_title_vectorized.toarray())

        # Add the PCA components to the input data
        input_data['PCA1'] = input_title_pca[:, 0]
        input_data['PCA2'] = input_title_pca[:, 1]

        # Use the trained KMeans model to predict the cluster
        predicted_cluster = self.kmeans.predict(input_title_vectorized)[0]

        return predicted_cluster

# Example usage:
# predictor = JobClusterPredictor(df)
# predictor.train_cluster_model()
# predicted_cluster = predictor.predict_cluster('Data Scientist', 'Mid-level', 'United States')
# print(f'Predicted Cluster: {predicted_cluster}')
