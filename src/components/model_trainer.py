import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {"n_estimators": [50, 100, 200]},
                "Decision Tree": {"max_depth": [None, 10, 20, 30]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "Linear Regression": {},  # No hyperparameters for Linear Regression
                "XGBRegressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            }

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            print("Model Report:", model_report)  # Debugging output

            if not model_report:
                raise CustomException("Model evaluation returned empty results.")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"Best Model: {best_model_name}, R² Score: {best_model_score}")  # Debugging output

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (R² < 0.6)")

            logging.info(f"Saving best model: {best_model_name}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)

            print("Actual Y Test:", y_test[:10])  # Debugging output
            print("Predicted Y:", predicted[:10])  # Debugging output

            r2_square = r2_score(y_test, predicted)

            print(f"Final R² Score: {r2_square:.4f}")  # Debugging output
            return r2_square

        except Exception as e:
            logging.error(f"Error in Model Trainer: {e}")
            raise CustomException(e, sys)
