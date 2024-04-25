import os
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, Response, jsonify
from utils import get_top_matches, get_similar_images


app = Flask(__name__)
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


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
