from src.components.data_ingestion import DataIngesion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class ComponentInit:
    def __init__(self):
        pass

    def inject_data(self):
        (raw_data,
         train_raw_data,
         test_raw_data) = DataIngesion().initiate_ingesion()

    def transform_train(self):
        (processed_input,
         target)=DataTransformation().initiate_data_transformation()
        ModelTrainer().initiate_trainer(processed_input, target)


