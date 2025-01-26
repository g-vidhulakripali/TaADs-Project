import configparser
import pickle
from sentence_transformers import SentenceTransformer
import torch
import sqlite3
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.env')

def get_torch_device():
    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    elif torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
    return torch_device

def fetch_records_in_batches(db_file, batch_size=5000):
    """
    Fetch records from the database in batches.
    :param db_file: database file location
    :param batch_size: number of records to fetch per batch
    :return: generator yielding batches of combined text
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    query_count = "SELECT COUNT(*) FROM zqm_module_en"
    cursor.execute(query_count)
    total_records = cursor.fetchone()[0]
    logging.info(f"Total records to process: {total_records}")

    for offset in range(0, total_records, batch_size):
        logging.info(f"Fetching records {offset} to {offset + batch_size}")
        query = f"""
        SELECT learning_obj, course_contents, prerequisites 
        FROM zqm_module_en
        LIMIT {batch_size} OFFSET {offset}
        """
        cursor.execute(query)
        records = cursor.fetchall()

        combined_records = [
            " ".join(filter(None, [
                row[0].strip() if isinstance(row[0], str) else None,
                row[1].strip() if isinstance(row[1], str) else None,
                row[2].strip() if isinstance(row[2], str) else None
            ])) for row in records
        ]

        yield [record for record in combined_records if record.strip()]

    conn.close()

def train_learning_obj_en(db, model_output, batch_size=5000, encode_batch_size=64):
    """
    Trains a model using SentenceTransformer and the description from the skills in the database.
    Processes records in batches to handle large datasets efficiently.
    @param db: path to database location
    @param model_output: output file containing the trained serialized language model.
    @param batch_size: number of records to fetch per database query batch
    @param encode_batch_size: number of records to encode in parallel per SentenceTransformer batch
    @return: None
    """
    logging.info("Loading the SentenceTransformer model.")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = get_torch_device()
    model.to(device)
    logging.info(f"Model loaded and moved to device: {device}.")

    try:
        with open(model_output, "wb") as fo:
            for batch_num, batch in enumerate(fetch_records_in_batches(db, batch_size), start=1):
                logging.info(f"Processing batch {batch_num} with {len(batch)} records.")

                try:
                    embeddings = model.encode(batch, batch_size=encode_batch_size, convert_to_tensor=True, device=device)
                    for embedding in embeddings:
                        pickle.dump(embedding.cpu().numpy(), fo)
                except Exception as e:
                    logging.error(f"Error processing batch {batch_num}: {e}")

        logging.info(f"Embeddings successfully saved to {model_output}.")
    except IOError as err:
        logging.error(f"IOError: {err}")

if __name__ == "__main__":
    logging.info("Starting the training process.")
    location_db = "../data/db/courses.sqlite"  # Path to the database
    model_vectorised_loc = "../data/models/all-MiniLM-L6-v2_embeddings_en.pkl"  # Output file for embeddings
    train_learning_obj_en(location_db, model_vectorised_loc)
    logging.info("Training process completed.")
