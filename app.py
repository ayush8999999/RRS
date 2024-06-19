from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

file_path = "C:\\Users\\ayush\\OneDrive\\Desktop\\rrs\\dataset.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Handle missing values by replacing them with an empty string
df['cuisines'] = df['cuisines'].fillna('')

# TF-IDF vectorization for food descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cuisines'])

# Calculate cosine similarity between food descriptions
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_cost = request.form.get('cost')
    user_food = request.form.get('food')
    user_location = request.form.get('location')

    # Filter dataset based on user preferences
    filtered_data = df[df['location'] == user_location]

    # If the filtered dataset is empty, recommend a random restaurant
    if filtered_data.empty:
        recommended_restaurant = df.sample(1)
    else:
        # Check if user food exists in the 'cuisines' column
        if user_food in df['food'].values:
            food_index = df.index[df['food'] == user_food].tolist()[0]  
            sim_scores = list(enumerate(cosine_sim[food_index]))

            # Sort restaurants based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get indices of top recommendations
            top_indices = [x[0] for x in sim_scores]

            # Exclude already filtered restaurants
            top_indices = [idx for idx in top_indices if idx in filtered_data.index]

            # If no matching recommendations, recommend a random restaurant from the filtered dataset
            if not top_indices:
                recommended_restaurant = filtered_data.sample(1)
            else:
                # Choose the first matching recommendation
                recommended_restaurant = df.loc[top_indices[0]]
        else:
            # Handle the case when user_food is not found in 'cuisines'
            recommended_restaurant = df.sample(1)

    return render_template('recommendation.html', recommendations=[recommended_restaurant.to_dict()])

if __name__ == '__main__':
    app.run(debug=True)