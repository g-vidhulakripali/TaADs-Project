import configparser
import pickle
from sentence_transformers import SentenceTransformer
import torch
from db_queries import get_learning_obj_en


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


def train_learning_obj_en(db, model_output):
    """
    Trains a LLM using SentenceTransformer and the description from the skills in the database.
    @param db: path to database location
    @param model_output: output file containing the trained serialized language model.
    @return: None
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = get_torch_device()
    model.to(device)

    # Retrieve learning objectives
    learning_obj = get_learning_obj_en(db)
    # Flatten the list and filter for valid, non-empty strings
    flat_learning_obj = [item for sublist in learning_obj for item in sublist if isinstance(item, str) and item.strip()]

    # Check for empty or invalid data
    if not flat_learning_obj:
        print("No valid learning objectives found. Exiting.")
        return

    try:
        vectors = []
        for text in flat_learning_obj:
            try:
                embedding = model.encode(text, convert_to_tensor=True, device=device)
                vectors.append(embedding.cpu().numpy())
            except Exception as e:
                print(f"Error encoding text '{text}': {e}")

        # Save the model embeddings to disk
        with open(model_output, "wb") as fo:
            pickle.dump(vectors, fo)
        print(f"Embeddings saved to {model_output}.")
    except IOError as err:
        print(f"IOError: {err}")


if __name__ == "__main__":
    # Demonstration of how to call the function
    location_db = "../data/db/courses.sqlite"  # Path to the database
    model_vectorised_loc = "../data/models/all-MiniLM-L6-v2_embeddings_en.pkl"  # Output file for embeddings
    # Train model in English
    train_learning_obj_en(location_db, model_vectorised_loc)
