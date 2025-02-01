import sys
import logging

def error_message_details(error, error_detail: sys):
    """Returns a detailed error message including filename and line number."""
    _, _, exc_tb = error_detail.exc_info()  # Get exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the script name
    error_message = "Error occurred in script [{0}], line [{1}], error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Corrected `super()` usage
        self.error_message = error_message_details(error_message, error_detail=error_detail)  # Fixed function call

    def __str__(self):
        return self.error_message
