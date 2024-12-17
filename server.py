import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from util.api_utils import get_learning_obj_en, load_faiss_index, store_in_faiss
from cachetools import cached, TTLCache
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import Ollama
import re
import difflib

# Configuration
logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME', 'paraphrase-MiniLM-L6-v2')  # Default model for vectorization
db_file = "data/db/courses.sqlite"
faiss_index_file = "data/faiss_index/faiss_index.idx"

# Load SentenceTransformer model for vectorization
logging.info("Loading SentenceTransformer model for vectorization...")
vector_model = SentenceTransformer(llm_name)

# Initialize the Ollama model using LangChain
logging.info("Initializing Ollama for generative responses...")
ollama_llm = Ollama(model="mistral")

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)

# Load records from the database
logging.info("Loading records from the database...")
records = get_learning_obj_en(db_file)
if not records:
    raise Exception("No records found in the database.")

# Function to combine course fields
def combine_course_fields_with_weights(course):
    """
    Combine course fields into a single string with weights for better relevance.
    """
    title_weight = 2
    learning_obj_weight = 2
    course_contents_weight = 1
    prerequisites_weight = 1

    title = (course[0] + " ") * title_weight
    learning_obj = (course[2] + " ") * learning_obj_weight
    course_contents = (course[3] + " ") * course_contents_weight
    prerequisites = (course[4] + " ") * prerequisites_weight

    return f"{title}{learning_obj}{course_contents}{prerequisites}"

def store_weighted_embeddings(records, faiss_index_file):
    """
    Generate weighted embeddings for courses and store them in the FAISS index.
    """
    logging.info("Generating weighted embeddings for courses...")
    weighted_texts = [combine_course_fields_with_weights(record) for record in records]
    embeddings = vector_model.encode(weighted_texts)

    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_index_file)
    logging.info("FAISS index with weighted embeddings stored successfully.")

# Generate and store weighted embeddings
store_weighted_embeddings(records, faiss_index_file)

# Load FAISS index
index = load_faiss_index(faiss_index_file)
if index is None or index.ntotal != len(records):
    raise Exception("FAISS index could not be loaded or does not match the number of records.")

# Vectorization with caching
@cached(vector_cache)
def cached_vectorize_input(user_input):
    """Cache vectorization to speed up repeated queries."""
    return vector_model.encode([user_input])[0]

def preprocess_query(query, vocabulary):
    """
    Preprocess the query by removing filler words, correcting common misspellings,
    and focusing on core terms.
    """
    query = re.sub(r"[^\w\s]", "", query)

    processed_words = []
    for word in query.lower().split():
        matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)
        if matches:
            processed_words.append(matches[0])
        else:
            processed_words.append(word)

    return " ".join(processed_words).strip()

def build_vocabulary(records):
    """
    Dynamically build a vocabulary from course titles, descriptions, and other terms in the records.
    """
    vocabulary = set()
    for record in records:
        for field in record:
            if isinstance(field, str):
                vocabulary.update(field.lower().split())
    return list(vocabulary)

@app.route('/search/', methods=['POST'])
async def search():
    """
    Unified API to search for relevant courses and provide a conversational response using Ollama.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Build a dynamic vocabulary
        vocabulary = build_vocabulary(records)

        # Preprocess the query
        processed_query = preprocess_query(query, vocabulary)
        logging.debug(f"Processed query: {processed_query}")

        # Vectorize the processed query
        user_vector = cached_vectorize_input(processed_query)

        # Perform FAISS search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=1)

        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Get the top result
        top_idx = indices[0][0]
        top_distance = D[0][0]
        threshold = 1.5

        if 0 <= top_idx < len(records) and top_distance < threshold:
            course = records[top_idx]
            response = ollama_llm(f"If you're interested in '{query}', you might enjoy the course '{course[0]}'.\n\nHere's a quick summary: {course[2]}\n\nPlease summarize this course in a conversational and engaging manner.")
            return jsonify({
                "result": {
                    "title": course[0],
                    "instructor": course[1],
                    "learning_obj": course[2],
                    "course_contents": course[3],
                    "prerequisites": course[4],
                    "credits": course[5],
                    "evaluation": course[6],
                    "time": course[7],
                    "frequency": course[8],
                    "duration": course[9],
                    "course_type": course[10]
                },
                "response": response
            }), 200

        return jsonify({
            "result": {},
            "response": "No relevant courses found for your query. Please refine your query or try a different topic."
        }), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)