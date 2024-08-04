from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('insurance_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Insurance Premium Prediction Page!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['Age'], data['Diabetes'], data['BloodPressureProblems'],
                data['AnyTransplants'], data['AnyChronicDiseases'], data['Height'],
                data['Weight'], data['KnownAllergies'], data['HistoryOfCancerInFamily'],
                data['NumberOfMajorSurgeries']]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'PremiumPrice': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
