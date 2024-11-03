import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from cachetools import cached, TTLCache
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for frequently accessed data
cache = TTLCache(maxsize=100, ttl=300)

# Function to vectorize learning objectives using a Sentence Transformer model
@cached(cache)
def vectorize_learning_obj(llm_name, learning_obj_tuple):
    model = SentenceTransformer(llm_name)
    vectors = model.encode(list(learning_obj_tuple))  # Convert back to list for encoding
    return vectors

# Function to store vectors in a Faiss index and save the index to a file
def store_in_faiss(records, faiss_index_file):
    try:
        # Create directory if it doesn't exist
        faiss_index_dir = os.path.dirname(faiss_index_file)
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)

        # Vectorize only the title field for Faiss index
        llm_name = "sentence-transformers/all-MiniLM-L6-v2"
        titles = [record[0] for record in records]  # Extract titles
        vectors = vectorize_learning_obj(llm_name, tuple(titles))  # Convert list to tuple for caching

        # Save the vectors to Faiss index
        dim = vectors.shape[1]  # Dimension of the vectors
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        # Save the index to a file
        faiss.write_index(index, faiss_index_file)
        logger.info(f"Vectors stored in Faiss index and saved to {faiss_index_file}")

    except Exception as e:
        logger.error(f"Error storing vectors in Faiss index: {e}")

# Function to fetch learning objectives from SQLite database
@cached(cache)
def get_learning_obj_en(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        statement = 'SELECT title, instructor, learning_obj, course_contents, prerequisites, credits, evaluation, time, frequency, duration, course_type FROM zqm_module_en'
        cursor.execute(statement)
        records = cursor.fetchall()
        conn.close()
        return records
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return []

# Function to load Faiss index
def load_faiss_index(faiss_index_file):
    try:
        index = faiss.read_index(faiss_index_file)
        return index
    except Exception as e:
        logger.error(f"Error loading Faiss index: {e}")

# Function to vectorize user input
@cached(cache)
def vectorize_input(llm_name, user_input):
    model = SentenceTransformer(llm_name)
    vector = model.encode([user_input])[0]
    return vector
