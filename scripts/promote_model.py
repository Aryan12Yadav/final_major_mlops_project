# promote model

import os
import mlflow

def promote_model():
    # Set up Dagshub credentials for MLflow tracking
    dagshub_token = os.getenv('CAPSTONE_TEST')
    if not dagshub_token:
        raise EnvironmentError('CAPTSTONE_TEST environment variable is not set')

    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    