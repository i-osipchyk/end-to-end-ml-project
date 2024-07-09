import os
import joblib
import pandas as pd
from sklearn.svm import SVC

from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(os.path.join(self.config.data_path, 'x_train.csv'))
        y_train = pd.read_csv(os.path.join(self.config.data_path, 'y_train.csv'))

        svc = SVC(gamma=0.1, C=10, random_state=42)

        svc.fit(X_train, y_train)
        joblib.dump(
            svc,
            os.path.join(
                self.config.root_dir,
                self.config.model_name
            )
        )