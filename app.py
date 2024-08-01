from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    data = pd.read_csv('data.csv', encoding='ISO-8859-1')
    data.dropna(inplace=True)
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data['InvoiceNo'] = data['InvoiceNo'].astype('str')
    data['StockCode'] = data['StockCode'].astype('str')
    scaler = StandardScaler()
    data[['Quantity', 'UnitPrice']] = scaler.fit_transform(data[['Quantity', 'UnitPrice']])
    return data

data = load_and_preprocess_data()

# Create user-product matrix
user_product_matrix = pd.pivot_table(
    data, 
    index='CustomerID', 
    columns='StockCode', 
    values='Quantity', 
    aggfunc='sum'
).fillna(0)

# Train SVD model
def train_model(data):
    reader = Reader(rating_scale=(0, 1))
    data_surprise = Dataset.load_from_df(data[['CustomerID', 'StockCode', 'Quantity']], reader)
    trainset, testset = train_test_split(data_surprise, test_size=0.25)
    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    algo.fit(trainset)
    return algo, testset

algo, testset = train_model(data)

# Routes
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id is None:
        return jsonify({"error": "user_id parameter is required"}), 400
    
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "user_id must be an integer"}), 400

    if user_id not in user_product_matrix.index:
        return jsonify({"error": "Invalid user_id"}), 400
    
    user_ratings = user_product_matrix.loc[user_id]
    unseen_products = user_ratings[user_ratings == 0].index.tolist()
    recommendations = [
        algo.predict(user_id, product) for product in unseen_products
    ]
    recommendations = sorted(recommendations, key=lambda x: x.est, reverse=True)[:10]
    recommended_products = [rec.iid for rec in recommendations]
    return jsonify(recommended_products)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.route('/')
def home():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
