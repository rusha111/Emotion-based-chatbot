import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('dataset/tweet_emotions.csv')
y = df['sentiment']
X = df['content']

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X, y)
joblib.dump(model, 'chatbot_model.pkl')
print("Model trained and saved successfully!")