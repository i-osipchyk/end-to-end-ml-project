from scipy.special import boxcox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.skewed_columns = ['chlorides', 'residual sugar', 'sulphates', 'total sulfur dioxide',
                               'free sulfur dioxide', 'fixed acidity', 'alcohol']
        self.boxcox_lambda = 0.05
        self.scaler = StandardScaler()
        self.oversampler = SMOTE(random_state=42)
        self.target = 'quality'

    def read_data(self) -> pd.DataFrame:
        logger.info("Reading the data")
        data = pd.read_csv(self.config.data_path)
        logger.info("Data was read successfully")
        return data

    def remove_skewness(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Removing skewness")

        for column in self.skewed_columns:
            data[column] = boxcox(data[column] + 1e-6, self.boxcox_lambda)

        logger.info("Removing skewness finished")

        return data

    def encode_labels(self, data: pd.DataFrame) -> pd.DataFrame:

        label_mapping = {
            3: 'bad',
            4: 'bad',
            5: 'fair',
            6: 'fair',
            7: 'good',
            8: 'good'
        }

        data[self.target] = data[self.target].map(label_mapping)

        encoder = LabelEncoder()
        data[self.target] = encoder.fit_transform(data[self.target])

        return data

    def train_test_split(self, data: pd.DataFrame) -> dict:

        logger.info("Splitting the data into train, val and test sets")

        x_train, x_temp, y_train, y_temp = train_test_split(
            data.drop('quality', axis=1), data['quality'],
            shuffle=True,
            stratify=data['quality'],
            test_size=0.3,
            random_state=42
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp,
            shuffle=True,
            stratify=y_temp,
            test_size=0.5,
            random_state=42
        )

        logger.info("Splitting finished")

        return {
            'x_train': x_train,
            'x_val': x_val,
            'x_test': x_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    def scale_data(self, data_dict: dict) -> dict:

        logger.info("Performing feature scaling")

        self.scaler.fit(data_dict['x_train'])
        data_dict['x_train'] = self.scaler.transform(data_dict['x_train'])
        data_dict['x_val'] = self.scaler.transform(data_dict['x_val'])
        data_dict['x_val'] = self.scaler.transform(data_dict['x_val'])

        logger.info("Feature scaling finished. Saving scaler")
        joblib.dump(self.scaler, os.path.join(self.config.root_dir, "scaler.pkl"))

        return data_dict

    def oversample(self, data_dict: dict) -> dict:
        logger.info(f"Current number of train samples: {data_dict['x_train'].shape[0]}. Performing oversampling")

        data_dict['x_train'], data_dict['y_train'] = self.oversampler.fit_resample(data_dict['x_train'],
                                                                                   data_dict['y_train'])

        logger.info(f"Oversampling finished. New number of train samples: {data_dict['x_train'].shape[0]}")

        return data_dict

    def save_data(self, data_dict: dict):
        for key, value in data_dict.items():
            pd.DataFrame(value).to_csv(os.path.join(self.config.root_dir, f"{key}.csv"), index=False)

        logger.info("Split data into train, val and test sets")

    def transform_data(self):
        data = self.read_data()
        data = self.remove_skewness(data)
        data = self.encode_labels(data)
        data_dict = self.train_test_split(data)
        data_dict = self.scale_data(data_dict)
        # data_dict = self.oversample(data_dict)
        self.save_data(data_dict)
