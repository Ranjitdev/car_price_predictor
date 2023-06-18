from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model
from dataclasses import dataclass
import os
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


@dataclass()
class ModelTrainerConfig:
    processed_data: str = os.path.join('artifacts', 'processed_data.csv')
    model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_trainer(self, processed_input, target):
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'XGBRegressor': XGBRegressor(),
            'CatBoostRegressor': CatBoostRegressor(verbose=False),
            'GradientBoostingRegressor': GradientBoostingRegressor()
        }
        params = {
            'LinearRegression': {},
            'Ridge': {'alpha': [0.1, 0.2, 0.5, 0.7, 1, 5, 10, 20]},
            'Lasso': {'alpha': [0.1, 0.2, 0.5, 0.7, 1, 5, 10, 20]},
            'KNeighborsRegressor': {
                'n_neighbors': [5, 7, 9, 11, 13, 15],
                # 'weights' : ['uniform','distance'],
                # 'metric' : ['minkowski','euclidean','manhattan'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'DecisionTreeRegressor': {
                # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter':['best','random'],
                'max_depth': range(2, 25, 1),
                'min_samples_split': range(2, 20, 1),
                'min_samples_leaf': range(1, 15, 1),
                # 'max_features': ['sqrt', 'log2']
            },
            'RandomForestRegressor': {
                # 'n_estimators':range(10, 100, 10),
                # 'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                'max_depth': range(3, 25, 1),
                'min_samples_split': range(2, 15, 1),
                'min_samples_leaf': range(1, 15, 1),
                # 'max_features': ['sqrt', 'log2']
            },
            'GradientBoostingRegressor': {
                # 'n_estimators':range(25, 500, 25),
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'criterion': ['friedman_mse', 'squared_error'],
                # 'max_depth':range(3, 25, 1),
                # 'min_samples_split':range(2, 15, 1),
                # 'min_samples_leaf':range(1, 15, 1),
                # 'learning_rate': [1,0.5,.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'max_features': ['sqrt', 'log2']
            },
            'XGBRegressor': {
                'n_estimators':range(25, 500, 25),
                'learning_rate': [1, 0.5, .1, .01, .05, .001]
            },
            'CatBoostRegressor': {
                'depth': [6, 8, 10],
                'learning_rate': [1, 0.5, .1, .01, .05, .001],
                'iterations': [30, 50, 100]
            }
        }
        try:
            best_model_name, best_model_score, best_param = evaluate_model(
                processed_input, target, models, params
            )
            model = models[best_model_name]
            model.set_params(**best_param)
            model.fit(processed_input, target)
            save_obj(
                path=self.config.model_path,
                obj=model
            )
            logging.info(f'Got the best model and saved successfully')
        except Exception as e:
            raise CustomException(e, sys)