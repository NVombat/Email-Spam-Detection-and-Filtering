from sklearn.feature_extraction.text import CountVectorizer
import pickle

model = pickle.load(open("spam_pred_model.pkl", 'rb'))
vocab = pickle.load(open("training_vocab.pkl", 'rb'))

email = ["Nah I don't think he goes to usf, he lives around here though"]
# email = ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"]
# email = ["Lol your always so convincing."]

cv = CountVectorizer(vocabulary=vocab)
email_features = cv.transform(email)

prediction = model.predict(email_features)

if prediction[0] == 0:
    print("SPAM MAIL")
else:
    print("HAM MAIL")