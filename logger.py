import os
import logging
import functools
import traceback
from datetime import datetime

# Ensure the logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_filename(custom_name=None):
    """Generate log filename with optional custom name."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{custom_name}_{timestamp}.log" if custom_name else f"{timestamp}.log"
    return os.path.join(LOG_DIR, filename)

def setup_logger(custom_name=None):
    """Set up logging with an optional custom filename."""
    log_filename = get_log_filename(custom_name)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def log_function_call(func):
    """Decorator to log function calls, execution time, return values, and exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        log_entry = f"Function: {func.__name__} | Args: {args} | Kwargs: {kwargs}"
        logging.info(log_entry)
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"Function: {func.__name__} | Returned: {result} | Execution time: {execution_time:.4f} sec \n")
            return result
        except Exception as e:
            error_message = f"Function: {func.__name__} | Exception: {str(e)}\n{traceback.format_exc()}\n"
            logging.error(error_message)
            raise  # Re-raise exception after logging it
    return wrapper

def log_class_methods(cls):
    """Class decorator to log all method calls in a class."""
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_function_call(attr_value))
    return cls
