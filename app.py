from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return "Machine Learning  Model Deployment using Flask"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["number"]
    prediction = model.predict(np.array([[data]]))
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)