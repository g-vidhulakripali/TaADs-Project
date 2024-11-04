import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from cachetools import cached, TTLCache

cache = TTLCache(maxsize=100, ttl=300)

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
        faiss_index_dir = os.path.dirname(faiss_index_file)
        if not os.path.exists(faiss_index_dir):
            os.makedirs(faiss_index_dir)
        llm_name = "sentence-transformers/all-MiniLM-L6-v2"
        titles = [record[0] for record in records]  # Extract titles
        vectors = vectorize_learning_obj(llm_name, tuple(titles))  # Convert list to tuple for caching
        dim = vectors.shape[1]  # Dimension of the vectors
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, faiss_index_file)

    except Exception as e:
        print(f"Error storing vectors in Faiss index: {e}")
        
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
