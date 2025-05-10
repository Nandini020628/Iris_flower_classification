from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('iris_model.pkl', 'rb'))

# Route for homepage with flower info and button to prediction
@app.route('/')
def home():
    return render_template('newh.html')

# Route for prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict
        prediction = model.predict(features)

        # Classes and images
        classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        predicted_class = classes[int(prediction[0])]

        flower_images = {
            'Iris-setosa': 'setosa.jpg',
            'Iris-versicolor': 'versicolor.jpg',
            'Iris-virginica': 'virginica.jpg'
        }
        image_file = flower_images.get(predicted_class, 'default.jpg')

        return render_template(
            'predict.html',
            prediction_text=f'Predicted Iris Species: {predicted_class}',
            image_file=image_file
        )

    return render_template('predict.html')  # for GET requests

if __name__ == '__main__':
    app.run(debug=True, port=5001)

