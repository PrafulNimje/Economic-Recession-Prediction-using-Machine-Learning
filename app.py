from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__,static_url_path='/static')

# Load the saved model
model = joblib.load('recession_model.joblib')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    
    # Convert the prediction to a float
    prediction = float(prediction[0])
    
    return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run(debug=True)
