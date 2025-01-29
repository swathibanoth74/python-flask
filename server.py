from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Sample dataset for training
data = {
    'url': ['http://phishing-site.com', 'https://legit-site.com', 'http://secure-site.com', 'http://phishing-site2.com'],
    'label': [1, 0, 0, 1]
}

df = pd.DataFrame(data)
df['url_length'] = df['url'].apply(len)
df['is_https'] = df['url'].apply(lambda x: 1 if 'https' in x else 0)

X = df[['url_length', 'is_https']]
y = df['label']

# Train models
dt_classifier = DecisionTreeClassifier().fit(X, y)
rf_classifier = RandomForestClassifier().fit(X, y)
nb_classifier = GaussianNB().fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json['url']
    url_length = len(url)
    is_https = 1 if 'https' in url else 0
    new_data = pd.DataFrame([[url_length, is_https]], columns=['url_length', 'is_https'])

    dt_pred = dt_classifier.predict(new_data)[0]
    rf_pred = rf_classifier.predict(new_data)[0]
    nb_pred = nb_classifier.predict(new_data)[0]

    result = {
        "Decision Tree": "Phishing" if dt_pred == 1 else "Legitimate",
        "Random Forest": "Phishing" if rf_pred == 1 else "Legitimate",
        "Naive Bayes": "Phishing" if nb_pred == 1 else "Legitimate"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
