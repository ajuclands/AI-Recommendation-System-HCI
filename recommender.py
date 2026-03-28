import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:

    def __init__(self, dataset_path):

        # Load dataset
        self.data = pd.read_csv(dataset_path)

        # Convert genres to vectors
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.genre_matrix = self.vectorizer.fit_transform(self.data["genre"])

        # Calculate similarity matrix
        self.similarity = cosine_similarity(self.genre_matrix)

    def recommend(self, movie_name, top_n=5):

        if movie_name not in self.data["movie"].values:
            return []

        idx = self.data[self.data["movie"] == movie_name].index[0]

        similarity_scores = list(enumerate(self.similarity[idx]))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []

        for i in similarity_scores[1:top_n + 1]:

            movie_data = self.data.iloc[i[0]]

            recommendations.append({
                "movie": movie_data["movie"],
                "genre": movie_data["genre"],
                "poster_url": movie_data["poster_url"],
                "rating": movie_data["rating"]
            })

        return recommendations