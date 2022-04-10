from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
import pickle
import os

templates_path = os.path.abspath("./templates")
app = Flask(__name__, template_folder=templates_path)
app.secret_key = "somekey"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/spam_pred', methods=['POST'])
def predict():
    if request.method == "POST":
        email = request.form["Email"]

        cv = CountVectorizer()

        email_features = cv.transform(email)

        prediction = model.predict(email_features)
