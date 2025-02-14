from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dictionary from the pickle file
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Access individual components from the loaded dictionary
knn = loaded_data['knn_model']
tfidf_matrix = loaded_data['tfidf_matrix']
indices = loaded_data['indices']
df_new = loaded_data['dataframe']

# Clean and normalize the 'cuisines' and 'locality' columns
df_new['cuisines'] = df_new['cuisines'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
df_new['locality'] = df_new['locality'].str.lower().str.strip()


# Recommendation function
def recommend_restaurants_knn(city=None, locality=None, min_cost=None, max_cost=None, cuisine=None):
    try:
        # Create a copy of the dataframe to apply filters
        neighbors_df = df_new.copy()

        # Apply city filter
        if city:
            neighbors_df = neighbors_df[neighbors_df['city'].str.lower() == city.lower()]

        # Apply locality filter
        if locality:
            neighbors_df = neighbors_df[neighbors_df['locality'].str.lower() == locality.lower()]

        # Apply cost filters
        if min_cost is not None:
            neighbors_df = neighbors_df[neighbors_df['average_cost_for_two'] >= min_cost]

        if max_cost is not None:
            neighbors_df = neighbors_df[neighbors_df['average_cost_for_two'] <= max_cost]

        # Apply cuisine filter
        if cuisine:
            cuisine_list = [c.strip().lower() for c in cuisine.split(",")] if isinstance(cuisine, str) else cuisine
            neighbors_df = neighbors_df[
                neighbors_df['cuisines'].apply(
                    lambda x: any(c in x for c in cuisine_list) if isinstance(x, str) else False)
            ]

        # Check if any restaurants remain after filtering
        if neighbors_df.empty:
            return {"message": "No recommendations found after applying filters."}

        # Sort by rating (highest to lowest) and select top 10
        neighbors_df = neighbors_df.sort_values(by='aggregate_rating', ascending=False).head(20)

        # Return the recommended restaurants' details
        return neighbors_df[
            ['name', 'address', 'locality', 'cuisines', 'average_cost_for_two', 'aggregate_rating']].to_dict(
            orient='records')

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        # Retrieve filters from the request
        city = request.args.get('city')
        locality = request.args.get('locality')
        min_cost = request.args.get('min_cost', type=float)
        max_cost = request.args.get('max_cost', type=float)
        cuisine = request.args.get('cuisine')

        # Get recommendations based on filters
        recommendations = recommend_restaurants_knn(
            city=city,
            locality=locality,
            min_cost=min_cost,
            max_cost=max_cost,
            cuisine=cuisine
        )



        return render_template('index.html', recommendations=recommendations)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
