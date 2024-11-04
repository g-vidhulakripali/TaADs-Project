import configparser
import pickle
from sentence_transformers import SentenceTransformer
import torch
from db.db_queries import get_learning_obj_en

# Load configuration
config = configparser.ConfigParser()
config.read('config.env')

# Load Hugging Face token
hf_token = config['DEFAULT'].get('HUGGINGFACE_API_KEY')


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
    Trains a LLM using SentenceTransformer and the description from the skills in the database
    @param db: path to database location
    @param model_output: output file containing the trained serialized language model
    @return: None
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    device = get_torch_device()
    model.to(device)

    # Retrieve learning objectives
    learning_obj = get_learning_obj_en(db)
    flat_learning_obj = [item for sublist in learning_obj for item in sublist]

    try:
        vectors = []
        for text in flat_learning_obj:
            embedding = model.encode(text, convert_to_tensor=True, device=device)
            vectors.append(embedding.cpu().numpy())

        # Save model to disk
        with open(model_output, "wb") as fo:
            pickle.dump(vectors, fo)
    except IOError as err:
        print(err)


if __name__ == "__main__":
    # Demonstration of how to call the vectorise text off-line
    location_db = "../data/db/courses.sqlite"
    model_vectorised_loc = "../data/models/all-MiniLM-L6-v2_embeddings_en.pkl"
    # Train model in English
    train_learning_obj_en(location_db, model_vectorised_loc)
