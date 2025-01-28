import os
import logging
from dotenv import load_dotenv
from src.logging_config import setup_logging
from src.db.db_utils import fetch_records_in_batches
from src.device.device_utils import get_torch_device
from sentence_transformers import SentenceTransformer
import pickle

# Setup logging
setup_logging()

# Load environment variables
load_dotenv("config/config.env")

def generate_embeddings(db_path, output_path, batch_size=5000, encode_batch_size=64):
    """
    Generates embeddings from database records and saves them as a .pkl file.

    Args:
        db_path (str): Path to the SQLite database file.
        output_path (str): Path to save the .pkl file.
        batch_size (int): Number of records to fetch per database query batch.
        encode_batch_size (int): Number of records to encode in parallel.
    """
    logging.info("Initializing SentenceTransformer for embedding generation.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = get_torch_device()
    model.to(device)
    logging.info(f"Model loaded and moved to device: {device}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "wb") as fo:
            for batch_num, batch in enumerate(fetch_records_in_batches(db_path, batch_size), start=1):
                logging.info(f"Processing batch {batch_num} with {len(batch)} records.")
                try:
                    embeddings = model.encode(batch, batch_size=encode_batch_size, convert_to_tensor=True, device=device)
                    for embedding in embeddings:
                        pickle.dump(embedding.cpu().numpy(), fo)
                except Exception as e:
                    logging.error(f"Error processing batch {batch_num}: {e}")
        logging.info(f"Embeddings successfully saved to {output_path}.")
    except IOError as e:
        logging.error(f"IOError while saving embeddings: {e}")
    logging.info("Embedding generation completed.")

if __name__ == "__main__":
    # Load paths from environment variables
    db_path = os.getenv("DB_PATH")
    output_path = os.getenv("OUTPUT_PATH")

    if not db_path or not output_path:
        logging.error("Database path or output path is not defined in the .env file.")
        raise ValueError("Please define DB_PATH and OUTPUT_PATH in the .env file.")

    generate_embeddings(db_path, output_path)
