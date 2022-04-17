# ML Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

df = pd.read_csv('datasets/mail_data.csv')
df['Category'] = df.Category.map({'ham': 1, 'spam':0})

# x_train, x_test, y_train, y_test = train_test_split(
#     df['Message'], df['Category'], test_size=0.2, random_state=1)

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(df['Message'])
# testing_data = count_vector.transform(x_test)

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data, df['Category'])

pickle.dump(naive_bayes, open('spam_pred_model.pkl','wb'))
pickle.dump(count_vector.vocabulary_, open("training_vocab.pkl", 'wb'))

email = ["Nah I don't think he goes to usf, he lives around here though"]
test = CountVectorizer(vocabulary=count_vector.vocabulary_)
email = test.transform(email)

print(naive_bayes.predict(email))