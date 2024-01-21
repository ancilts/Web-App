from flask import Flask,render_template,request
import pickle

import numpy as np

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        model = pickle.load(open('model.pkl', 'rb'))

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)


        species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species_mapping[prediction[0]]

    return render_template('predict.html', species=predicted_species)



if __name__ == '__main__':
    app.run()