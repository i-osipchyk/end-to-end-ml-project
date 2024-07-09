import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib

from mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_uri)

    def eval_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    def log_into_mlflow(self):
        X_test = pd.read_csv(os.path.join(self.config.data_path, 'x_test.csv'), skiprows=1, header=None)
        y_test = pd.read_csv(os.path.join(self.config.data_path, 'y_test.csv'))

        model = joblib.load(self.config.model_path)

        with mlflow.start_run():
            y_pred = model.predict(X_test)

            acc, pr, rec, f1 = self.eval_metrics(y_test, y_pred)

            # Log metrics to mlflow
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('precision', pr)
            mlflow.log_metric('recall', rec)
            mlflow.log_metric('f1 score', f1)

            # Log parameters to mlflow
            mlflow.log_params(self.config.all_params)

            # Save metrics to a JSON file using save_json utility
            scores = {
                'accuracy': acc,
                'precision': pr,
                'recall': rec,
                'f1 score': f1
            }
            save_json(path=Path(os.path.join(self.config.root_dir, self.config.metric_file_name)), data=scores)

            # Log the model to mlflow
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, 'model', registered_model_name='SVC')
            else:
                mlflow.sklearn.log_model(model, "model")
