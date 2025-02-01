import logging
import os
from datetime import datetime

# Corrected the file name and log path creation
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")

# Ensure the "logs" directory exists, not the file
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Fixed typo in the format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s-%(message)s',  # Corrected 'actime' to 'asctime'
    level=logging.INFO,
)

         



