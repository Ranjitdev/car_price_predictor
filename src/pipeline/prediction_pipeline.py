import pandas as pd

from src.utils import load_obj
from src.utils import save_obj
from src.exception import CustomException
from src.logger import logging
import os
import sys
from dataclasses import dataclass


@dataclass()
class PredictPipeConfig:
    model: str = os.path.join('artifacts', 'model.pkl')
    preprocessor: str = os.path.join('artifacts', 'preprocessor.pkl')
    data: str = os.path.join('artifacts', 'raw_data.csv')

class PredictPipeline:
    def __init__(
            self,
            name=None,
            location=None,
            kms_driven=None,
            fuel_type=None,
            owner=None,
            year=None
    ):
        self.config = PredictPipeConfig
        self.Name = name
        self.Location = location
        self.Kms_driven = kms_driven
        self.Fuel_type = fuel_type
        self.Owner = owner
        self.Year = year

    def data_to_df(self):
        data_array = {
            'Name': self.Name,
            'Location': self.Location,
            'Kms_driven':self.Kms_driven,
            'Fuel_type': self.Fuel_type,
            'Owner': self.Owner,
            'Year': self.Year
        }
        data_df = pd.DataFrame([data_array])
        return data_df

    def predict_result(self, df):
        preprocessor = load_obj(
            self.config.preprocessor
        )
        model = load_obj(
            self.config.model
        )
        test_array = preprocessor.transform(df)
        prediction = model.predict(test_array)[0]
        return prediction