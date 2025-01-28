import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/config.env")

def setup_logging():
    """
    Configures logging for the application.
    Reads log file path and log level from environment variables.
    """
    # Load log file and log level from .env
    log_file = os.getenv("LOG_FILE", "logs/application.log")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Add a console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
