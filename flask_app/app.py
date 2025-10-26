from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ---------------- Text preprocessing ---------------- #
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in text.split() if w not in stop_words])

def removing_numbers(text):
    return ''.join([c for c in text if not c.isdigit()])

def lower_case(text):
    return " ".join([w.lower() for w in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    return re.sub('\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# ---------------- MLflow / Dagshub setup ---------------- #
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token and os.getenv("CI") != "true":
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = 'aryanyadav892408'
repo_name = 'final_major_mlops_project'
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ---------------- Flask app ---------------- #
app = Flask(__name__)
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions", ["prediction"], registry=registry
)

# ---------------- Load model & vectorizer ---------------- #
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        raise RuntimeError(f"No model versions found for '{model_name}'")
    return latest_version[0].version

if os.getenv("CI") == "true":
    # Dummy model for CI/CD tests
    class DummyModel:
        def predict(self, X):
            return ["Positive"] * len(X)
    model = DummyModel()
    vectorizer = pickle.loads(pickle.dumps({"dummy": True}))
else:
    model_version = get_latest_model_version(model_name)
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# ---------------- Routes ---------------- #
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    text = normalize_text(request.form["text"])

    if os.getenv("CI") == "true":
        prediction = model.predict([text])[0]
    else:
        features = vectorizer.transform([text])
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        prediction = model.predict(features_df)[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
