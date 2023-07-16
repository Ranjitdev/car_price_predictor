from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
import os
import dill


def save_obj(path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            dill.dump(obj=obj, file=file)
        logging.info(f'Saved the file {str(obj)}')
    except Exception as e:
        raise CustomException(e, sys)


def load_obj(path):
    try:
        with open(path, 'rb') as file:
            logging.info('Loaded the file')
            return dill.load(file=file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(processed_input, target, models, params):
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            processed_input, target, train_size=0.3, random_state=41
        )
        report = {}
        for i in models.keys():
            model = models[i]
            param = params[i]
            gs = GridSearchCV(
                model, param,
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1,
                verbose=2
            )
            gs.fit(processed_input, target)
            model.set_params(**gs.best_params_)

            model.fit(processed_input, target)
            pred_train = model.predict(x_train)
            pred_test = model.predict(x_test)
            train_score = np.round(r2_score(y_train, pred_train)*100, 2)
            test_score = np.round(r2_score(y_test, pred_test)*100, 2)
            report[i] = {
                'Train Score': train_score,
                'Test Score': test_score,
                'Best Param': gs.best_params_
            }
        reports = pd.DataFrame(report)
        best_model_score = max(reports.iloc[1, :])
        best_model_name = [i for i in reports.columns if reports[i][1] == best_model_score][0]
        best_param = [reports[i][2] for i in reports.columns if reports[i][1] == best_model_score][0]
        report_file = os.path.join('artifacts', 'training_report.csv')
        reports.to_csv(report_file)
        print(best_model_name, best_model_score)
        print(best_param)
        logging.info(
            f'{best_model_name} is selected with score {best_model_score} and parameters {best_param}'
        )

        return best_model_name, best_model_score, best_param
    except Exception as e:
        raise CustomException(e, sys)


@dataclass()
class StreamlitConfig:
    data: str = os.path.join('artifacts', 'raw_data.csv')


class StreamliDataProvider:
    def __init__(self):
        self.config = StreamlitConfig
        self.data = pd.read_csv(StreamlitConfig.data)
        self.car_names = self.data['Name'].unique()
        self.locations = self.data['Location'].unique()
        self.fuels = self.data['Fuel_type'].unique()
        self.owners = sorted(self.data['Owner'].unique())
        self.dates = sorted(self.data['Year'].unique())