import numpy as np
import pandas as pd
from flask import Flask, request, Response
import urllib.parse
from utils import get_top_matches


app = Flask(__name__)

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


@app.route('/')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
