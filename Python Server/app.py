import os
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from utils import get_top_matches, get_similar_images
app = Flask(__name__)
CORS(app, origins='*')
app.config['UPLOAD_FOLDER'] = './cache/images'

# Load Myntra Dataset
myntra = pd.read_csv('assets/myntra.csv')

filtered_indices = pd.read_pickle(r'assets/filtered_indices.pkl')
filtered_indices = np.array(filtered_indices)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    top_matches = get_top_matches(query)
    result_dict = {title: float(score) for title, score in top_matches}
    # print(result_dict)

    matching_records = myntra[myntra['corpusData'].isin(result_dict.keys())]
    matching_records['score'] = matching_records['corpusData'].map(result_dict)
    matching_records = matching_records.sort_values(by='score', ascending=False)
    matching_records_json = matching_records.to_json(orient='records')
    # print(matching_records_json)

    return Response(matching_records_json, mimetype='application/json')


myntra1 = pd.read_csv('assets/myntra.csv', usecols=['title', 'img1', 'product_id'])


@app.route('/shop', methods=['GET'])
def route_shop():
    # Get the page number from the request arguments (default to 1 if not provided)
    page = int(request.args.get('page', 1))
    # Calculate the start and end indices for the products to display on this page
    start_idx = (page - 1) * 10
    end_idx = min(start_idx + 10, len(myntra1))
    # Extract the URLs and titles for the products on this page
    products_on_page = myntra1.iloc[start_idx:end_idx]
    products_data = [{"imageUrl": row['img1'], "title": row['title'], "product_id": row['product_id']} for index, row in products_on_page.iterrows()]
    # Return the list of products data as JSON data along with the page number
    return jsonify({"products": products_data, "page": page})


@app.route('/product', methods=['GET'])
def get_product():
    product_id = int(request.args.get('product_id', ''))
    product = myntra[myntra['product_id'] == product_id]
    if product.empty:
        return jsonify({'error': 'Product not found'}), 404
    product_json = product.to_json(orient='records')
    return Response(product_json, mimetype='application/json')


@app.route('/recommend', methods=['GET'])
def get_recommend():
    title = request.args.get('title', '')

    # print(np.where(myntra['title'] == title))
    index = np.where(myntra['title'] == title)[0][0]
    output = filtered_indices[index][1:]
    # print(output)

    recommend_records_json = myntra.iloc[output].to_json(orient='records')
    return Response(recommend_records_json, mimetype='application/json')


@app.route('/imgsearch', methods=['POST'])
def reverse_img_search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], datetime.now().strftime('%Y%m%d%H%M%S') + '_' + image.filename)
    image.save(path)
    image_data = Image.open(path)

    similar_product_ids = get_similar_images(image_data)
    similar_products = myntra[myntra['product_id'].isin(similar_product_ids)]
    # Set product_id as the index
    similar_products = similar_products.set_index('product_id')
    # Reorder based on the order of similar_product_ids
    similar_products = similar_products.loc[similar_product_ids]
    similar_products_json = similar_products.reset_index().to_json(orient='records')
    return Response(similar_products_json, mimetype='application/json')


@app.route('/trending', methods=['GET'])
def get_trending():
    top_rating = myntra.sort_values(by='AverageRatingCount', ascending=False).head(10)
    trending_products_json = top_rating.to_json(orient='records')
    return Response(trending_products_json, mimetype='application/json')


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
