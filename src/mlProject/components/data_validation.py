import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.unzip_data_dir)
            all_data_cols = set(data.columns)

            all_schema = self.config.all_schema
            all_schema_cols = set(all_schema.keys())

            if all_data_cols != all_schema_cols:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                return validation_status

            for col, expected_dtype in all_schema.items():
                actual_dtype = data[col].dtype
                if str(actual_dtype) != expected_dtype:
                    validation_status = False
                    break

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e