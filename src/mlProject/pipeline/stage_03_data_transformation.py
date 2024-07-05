from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger


STAGE_NAME = "Data Transformation Stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.transform_data()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
