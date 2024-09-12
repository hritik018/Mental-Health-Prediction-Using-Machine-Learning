from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        try:
            gender = request.form['gender']
            happiness = request.form['happiness']
            sadness = request.form['sadness']
            anxiety = request.form['anxiety']
            worrying = request.form['worrying']

            # Convert form data to integers
            inputs = [int(gender), int(happiness), int(sadness), int(anxiety), int(worrying)]

            # Pad the input with zeros to match the expected number of features
            required_features = 21
            if len(inputs) < required_features:
                inputs.extend([0] * (required_features - len(inputs)))

            # Reshape into numpy array
            arr = np.array(inputs).reshape(1, -1)

            # Make prediction using the loaded model
            pred = model.predict(arr)

            return render_template('predict.html', prediction=pred)
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
