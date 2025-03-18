from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Vectorizer
model = pickle.load(open('../models/logistic_model.pkl', 'rb'))
tfidf = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    transformed_text = tfidf.transform([news_text])
    prediction = model.predict(transformed_text)[0]
    result = 'Fake' if prediction == 1 else 'Real'
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
