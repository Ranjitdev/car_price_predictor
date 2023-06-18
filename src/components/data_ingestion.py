import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
import os
from sklearn.model_selection import train_test_split
from src.utils import save_obj


@dataclass
class DataIngesionConfig:
    raw_data: str = os.path.join('artifacts', 'raw_data.csv')
    train_df: str = os.path.join('artifacts', 'train.csv')
    test_df: str = os.path.join('artifacts', 'test.csv')


class DataIngesion:
    def __init__(self):
        self.config = DataIngesionConfig

    def initiate_ingesion(self):
        logging.info('Ingesion started')
        try:
            data = pd.read_csv('data.csv')
            train_df, test_df = train_test_split(data, test_size=0.3, random_state=41)
            os.makedirs(os.path.dirname(self.config.raw_data), exist_ok=True)
            data.to_csv(self.config.raw_data, header=True, index=False)
            train_df.to_csv(self.config.train_df, header=True, index=None)
            test_df.to_csv(self.config.test_df, header=True, index=None)
            logging.info('Data Ingesion completed')
            return data, train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)