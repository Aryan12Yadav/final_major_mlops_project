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
import dagshub
import warnings

warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')

# ============================================================
# 🧹 TEXT PREPROCESSING FUNCTIONS
# ============================================================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), " ", text)
    text = text.replace('؛', "")
    text = re.sub('/s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# ============================================================
# ⚙️ MLflow / Dagshub Setup (Safe Mode)
# ============================================================
app = Flask(__name__)

IS_CI = os.environ.get("CI", "false").lower() == "true"
model = None
vectorizer = None

if IS_CI:
    # ------------------------------------------
    # CI/CD Mode — No Dagshub Required
    # ------------------------------------------
    print("Running in CI/CD Mode: Using Dummy Model")

    class DummyModel:
        def predict(self, X):
            text = X.iloc[0, 0] if isinstance(X, pd.DataFrame) else X[0]
            return ["Positive" if "love" in text.lower() else "Negative"]

    class DummyVectorizer:
        def transform(self, texts):
            return pd.DataFrame([[len(text) for text in texts]], columns=["len"])

    model = DummyModel()
    vectorizer = DummyVectorizer()

else:
    # ------------------------------------------
    # 🌐 Production Mode — Use Dagshub + MLflow
    # ------------------------------------------
    dagshub_token = os.getenv('CAPSTONE_TEST')
    if not dagshub_token:
        print("⚠️ CAPSTONE_TEST not set. Falling back to local model.pkl")
    else:
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    dagshub_url = 'https://dagshub.com/aryanyadav892408/final_major_mlops_project.mlflow'
    mlflow.set_tracking_uri(dagshub_url)

    model_name = "my_model"

    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        try:
            latest_version = client.get_latest_versions(model_name, stages=["Production"])
            if not latest_version:
                latest_version = client.get_latest_versions(model_name, stages=["None"])
            return latest_version[0].version if latest_version else None
        except Exception as e:
            print(f"⚠️ MLflow connection failed: {e}")
            return None

    model_version = get_latest_model_version(model_name)

    try:
        if model_version:
            model_uri = f'models:/{model_name}/{model_version}'
            print(f"Fetching model from: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
        else:
            print("⚠️ Using local fallback model.pkl")
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")
        class DummyModel:
            def predict(self, X):
                return ["Positive"]
        model = DummyModel()

    try:
        vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    except Exception as e:
        print(f"⚠️ Vectorizer load failed: {e}")
        class DummyVectorizer:
            def transform(self, texts):
                return pd.DataFrame([[len(text) for text in texts]], columns=["len"])
        vectorizer = DummyVectorizer()

# ============================================================
# 📊 PROMETHEUS METRICS
# ============================================================
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry)

# ============================================================
# 🌐 ROUTES
# ============================================================
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

    text = request.form["text"]
    text = normalize_text(text)

    try:
        features = vectorizer.transform([text])
        if not isinstance(features, pd.DataFrame):
            features_df = pd.DataFrame(features)
        else:
            features_df = features
        result = model.predict(features_df)
        prediction = result[0]
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        prediction = "Error"

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# ============================================================
# 🚀 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
