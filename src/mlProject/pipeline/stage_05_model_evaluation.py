from mlProject import logger
from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_evaluation_config()
            model_eval_config = ModelEvaluation(config=model_eval_config)
            model_eval_config.log_into_mlflow()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        raise e
