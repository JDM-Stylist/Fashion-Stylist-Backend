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

myntra1 = pd.read_csv('assets/myntra.csv', usecols=['title', 'img1'])

@app.route('/shop', methods=['GET'])
def route_shop():
    if request.method == 'GET':
        page = int(request.args.get('page', 1))
        items_per_page = 10
        start_index = (page - 1) * items_per_page
        end_index = start_index + items_per_page

        # Filter and paginate the data
        paginated_data = myntra1.iloc[start_index:end_index]

        response_data = {
            'productsData': paginated_data.to_dict(orient='records'),
            'totalPages': len(myntra) // items_per_page + 1
        }

        return jsonify(response_data)


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
