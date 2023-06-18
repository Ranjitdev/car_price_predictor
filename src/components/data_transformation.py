import pandas as pd
import numpy as np
from src.exception import CustomException
from src.exception import logging
from src.utils import save_obj
import sys
import os
from dataclasses import dataclass
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


@dataclass
class ConfigTransformation:
    data: str = os.path.join('artifacts', 'raw_data.csv')
    train_df: str = os.path.join('artifacts', 'train.csv')
    test_df: str = os.path.join('artifacts', 'test.df')
    preprocessor: str = os.path.join('artifacts', 'preprocessor.pkl')
    processed_data: str = os.path.join('artifacts', 'processed_data.csv')


class DataTransformation:
    def __init__(self):
        self.transformation_config = ConfigTransformation

    def transform_pipe(self):
        numeric_features = ['Kms_driven', 'Owner', 'Year']
        categorical_features = ['Name', 'Location', 'Fuel_type']
        num_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )
        cat_pipe = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()
        )
        transformer_obj = ColumnTransformer([
            ('numerical', num_pipe, numeric_features),
            ('categorical', cat_pipe, categorical_features)
        ])
        return transformer_obj

    def initiate_data_transformation(self):
        try:
            data = pd.read_csv(self.transformation_config.data)
            target_column = 'Price'
            input_df = data.drop(target_column, axis=1)
            target_df = data[target_column]
            transformer_obj = self.transform_pipe()

            logging.info('Applying preprocessing algorithm on datasets')
            processed_array = transformer_obj.fit_transform(input_df).toarray()
            save_obj(
                path=self.transformation_config.preprocessor,
                obj=transformer_obj
            )
            logging.info('Saved preprocessed pipeline')

            transformed_df = pd.DataFrame(
                np.column_stack((processed_array, data[[target_column]]))
            )
            transformed_df.to_csv(
                self.transformation_config.processed_data, index=False, header=True
            )
            logging.info('Saved processed data')

            input_data = transformed_df.iloc[:, :-1]
            target_data = transformed_df.iloc[:, -1]
            return input_data, target_data
        except Exception as e:
            raise CustomException(e, sys)
