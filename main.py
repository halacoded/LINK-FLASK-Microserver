from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

model = joblib.load('tuned_xgb.joblib')

#endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json['data']
        prediction = model.predict([input_data])[0]
        return jsonify({'prediction': int(prediction)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.json["data"]
    predictions = [int(model.predict([row])[0]) for row in data]
    return jsonify({ "predictions": predictions })


#run Flask
if __name__ == '__main__':
    app.run(debug=True)