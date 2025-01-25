import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from cachetools import cached, TTLCache
import pickle
import numpy as np
from trainer import train_learning_obj_en

cache = TTLCache(maxsize=100, ttl=300)
location_db = "data/db/courses.sqlite"
model_vectorised_loc = "data/models/all-MiniLM-L6-v2_embeddings_en.pkl"
faiss_index_file = "data/faiss_index/faiss_index.idx"

@cached(cache)
def vectorize_learning_obj(llm_name, learning_obj_tuple):
    try:
        model = SentenceTransformer(llm_name)
        print(f"Model '{llm_name}' is pretrained.")
    except Exception as e:
        print(f"Model '{llm_name}' is not trained. Training or loading required.")
        raise e
    vectors = model.encode(list(learning_obj_tuple))  # Convert back to list for encoding
    return vectors

def store_in_faiss(records, faiss_index_file):
    try:
        # Check if the pickle file with precomputed embeddings exists
        if os.path.exists(model_vectorised_loc):
            with open(model_vectorised_loc, "rb") as f:
                vectors = pickle.load(f)
            print("Loaded precomputed embeddings from pickle file.")
        else:
            # Generate embeddings and save them if pickle file doesn't exist
            train_learning_obj_en(location_db, model_vectorised_loc)
            with open(model_vectorised_loc, "rb") as f:
                vectors = pickle.load(f)
            print("Generated and saved embeddings to pickle file.")

        faiss_index_dir = os.path.dirname(faiss_index_file)
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)
        llm_name = "sentence-transformers/all-MiniLM-L6-v2"
        titles = [record[0] for record in records]  # Extract titles
        vectors = vectorize_learning_obj(llm_name, tuple(titles))

        # Store vectors in FAISS index
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, faiss_index_file)
        print("Stored vectors in FAISS index.")

    except Exception as e:
        print(f"Error storing vectors in Faiss index: {e}")

@cached(cache)
def get_learning_obj_en(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        statement = 'SELECT title, instructor, learning_obj, course_contents, prerequisites, credits, evaluation, time, frequency, duration, course_type, platform FROM zqm_module_en'
        cursor.execute(statement)
        records = cursor.fetchall()
        conn.close()
        return records
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []

def load_faiss_index(faiss_index_file):
    try:
        index = faiss.read_index(faiss_index_file)
        return index
    except Exception as e:
        print(f"Error loading Faiss index: {e}")

@cached(cache)
def vectorize_input(llm_name, user_input):
    model = SentenceTransformer(llm_name)
    vector = model.encode([user_input])[0]
    return vector
