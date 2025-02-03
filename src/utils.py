import os
import sys
import dill  # Using dill instead of pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """Save an object using dill."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Using dill instead of pickle
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Train models, evaluate them, and return R² scores."""
    try:
        report = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")  # Debugging output

            if model_name in params and params[model_name]:  # Check if parameters exist
                gs = GridSearchCV(model, params[model_name], cv=3, scoring="r2", n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                print(f"Best Params for {model_name}: {gs.best_params_}")  # Debugging output

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"{model_name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")  # Debugging output

            report[model_name] = test_model_score

        print("Final Model Report:", report)  # Debugging output

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)  # Using dill instead of pickle
    except Exception as e:
        raise CustomException(e, sys)
