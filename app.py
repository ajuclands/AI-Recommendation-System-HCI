from flask import Flask, render_template, request
from recommender import RecommendationEngine

app = Flask(__name__)

# Initialize recommendation engine
engine = RecommendationEngine("dataset.csv")


@app.route("/")
def home():
    # Get top rated movies for homepage
    top_movies = engine.data.sort_values(by="rating", ascending=False).head(5).to_dict('records')
    return render_template("index.html", top_movies=top_movies)


@app.route("/recommend", methods=["POST"])
def recommend():

    movie = request.form.get("movie")
    genre_filter = request.form.get("genre")

    results = engine.recommend(movie)

    # Optional genre filtering
    if genre_filter:
        results = [r for r in results if genre_filter.lower() in r["genre"].lower()]

    return render_template("result.html", movie=movie, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5050)