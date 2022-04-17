from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
import pickle
import os


templates_path = os.path.abspath("./templates")
app = Flask(__name__, template_folder=templates_path)
app.secret_key = "somekey"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


model = pickle.load(open('./ml/spam_pred_model.pkl', 'rb'))
vocab = pickle.load(open("./ml/training_vocab.pkl", 'rb'))


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        email_form = request.form["Email"]

        email = []
        email.append(email_form)

        cv = CountVectorizer(vocabulary=vocab)
        email_features = cv.transform(email)

        prediction = model.predict(email_features)

        if prediction[0] == 0:
            print("SPAM MAIL")
            return render_template("spampred.html", error="THIS LOOKS LIKE A SPAM MAIL")
        else:
            print("HAM MAIL")
            return render_template("spampred.html", error="THIS DOES NOT LOOK LIKE A SPAM MAIL")

    else:
        return render_template("spampred.html")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
