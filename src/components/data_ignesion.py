import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transforrmation import DataTransformation
from src.components.data_transforrmation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig

from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            dataset_path = 'notebook/data/stud.csv'  # Path to dataset
            
            # Check if the dataset file exists
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
            
            df = pd.read_csv(dataset_path)
            logging.info("Dataset successfully loaded into dataframe")

            # Create necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Initialize and perform data ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Initialize and perform data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Initialize and train the model
        model_trainer = ModelTrainer()
        model_training_result = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Log the model training result
        logging.info(f"Model Training Result: {model_training_result}")

    except Exception as e:
        logging.error(f"Error occurred during the pipeline execution: {str(e)}")
        sys.exit(1)
