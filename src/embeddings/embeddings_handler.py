import logging
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def combine_course_fields_with_weights(course):
    """
    Combine course fields into a single string with weights for better relevance.
    """
    try:
        title_weight = 2
        learning_obj_weight = 2
        course_contents_weight = 1
        prerequisites_weight = 1

        # Safely handle None values
        title = (str(course[0] or "") + " ") * title_weight
        learning_obj = (str(course[2] or "") + " ") * learning_obj_weight
        course_contents = (str(course[3] or "") + " ") * course_contents_weight
        prerequisites = (str(course[4] or "") + " ") * prerequisites_weight

        combined_text = f"{title}{learning_obj}{course_contents}{prerequisites}"
        # logging.debug(f"Combined fields for record: {combined_text[:50]}...")  # Log first 50 chars for brevity
        return combined_text
    except Exception as e:
        logging.error(f"Error combining course fields: {e}. Record: {course}")
        return None


def store_embeddings_and_faiss(records, faiss_index_file, vector_model):
    """
    Generate weighted embeddings for courses and store them in the FAISS index.
    """
    try:
        logging.info("Generating weighted embeddings for courses...")
        weighted_texts = [combine_course_fields_with_weights(record) for record in records if record]
        weighted_texts = [text for text in weighted_texts if text]  # Skip None values
        logging.info(f"Total valid records for embeddings: {len(weighted_texts)}")

        embeddings = vector_model.encode(weighted_texts)
        logging.debug(f"Embeddings shape: {embeddings.shape}")

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, faiss_index_file)
        logging.info(f"FAISS index with weighted embeddings stored successfully at {faiss_index_file}.")
    except Exception as e:
        logging.error(f"Error during FAISS index creation: {e}")


def store_embeddings(records, faiss_index_file, pkl_file, vector_model):
    """
    Generate embeddings for the given records, store them in a .pkl file,
    and create a FAISS index.

    Args:
        records (list): List of course records.
        faiss_index_file (str): Path to the FAISS index file.
        pkl_file (str): Path to the pickle file to store embeddings.
        vector_model: Pretrained SentenceTransformer model.
    """
    try:
        # Check if pickle file exists
        if os.path.exists(pkl_file):
            logging.info(f"Precomputed embeddings file found at {pkl_file}. Loading embeddings...")
            with open(pkl_file, "rb") as f:
                embeddings = pickle.load(f)
            logging.info(f"Embeddings successfully loaded from {pkl_file}.")
        else:
            logging.info(f"No precomputed embeddings file found. Generating embeddings...")
            # Generate weighted texts
            weighted_texts = [combine_course_fields_with_weights(record) for record in records if record]
            weighted_texts = [text for text in weighted_texts if text]  # Skip None values
            logging.info(f"Total valid records for embeddings: {len(weighted_texts)}")

            embeddings = vector_model.encode(weighted_texts)
            logging.debug(f"Embeddings shape: {embeddings.shape}")

            # Save embeddings to .pkl file
            with open(pkl_file, "wb") as f:
                pickle.dump(embeddings, f)
            logging.info(f"Generated embeddings and saved to {pkl_file}.")

        # Create FAISS index directory if it doesn't exist
        faiss_index_dir = os.path.dirname(faiss_index_file)
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)

        # Create and store FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, faiss_index_file)
        logging.info(f"FAISS index successfully created and stored at {faiss_index_file}.")
    except Exception as e:
        logging.error(f"Error during embedding generation or FAISS index creation: {e}")


def load_faiss_index(faiss_index_file):
    """
    Load the FAISS index from the specified file.

    Args:
        faiss_index_file (str): Path to the FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index, or None if the file does not exist.
    """
    try:
        if os.path.exists(faiss_index_file):
            logging.info(f"Loading FAISS index from {faiss_index_file}...")
            index = faiss.read_index(faiss_index_file)
            logging.info(f"FAISS index successfully loaded with {index.ntotal} entries.")
            return index
        else:
            logging.error(f"FAISS index file not found at {faiss_index_file}.")
            return None
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        return None
