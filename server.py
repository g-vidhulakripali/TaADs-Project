import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from api_utils import get_learning_obj_en, load_faiss_index, store_in_faiss
from cachetools import cached, TTLCache
from sentence_transformers import SentenceTransformer
import faiss
import requests
import re
import difflib
import os
from dotenv import load_dotenv

# Configuration
logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.env')

# Load environment variables from config.env
load_dotenv("config.env")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME', 'all-mpnet-base-v2')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Load SentenceTransformer model for vectorization
logging.info("Loading SentenceTransformer model for vectorization...")
vector_model = SentenceTransformer(llm_name)

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)

# Load records from the database
logging.info("Loading records from the database...")
records = get_learning_obj_en("data/db/courses.sqlite")
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

    title = (str(course[0] or "") + " ") * title_weight
    learning_obj = (str(course[2] or "") + " ") * learning_obj_weight
    course_contents = (str(course[3] or "") + " ") * course_contents_weight
    prerequisites = (str(course[4] or "") + " ") * prerequisites_weight

    return f"{title}{learning_obj}{course_contents}{prerequisites}"

def store_weighted_embeddings(records, faiss_index_file):
    """
    Generate weighted embeddings for courses and store them in the FAISS index.
    """
    logging.info("Generating weighted embeddings for courses...")
    weighted_texts = [combine_course_fields_with_weights(record) for record in records]
    embeddings = vector_model.encode(weighted_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_index_file)
    logging.info("FAISS index with weighted embeddings stored successfully.")

# Generate and store weighted embeddings
store_weighted_embeddings(records, "data/faiss_index/faiss_index.idx")

# Load FAISS index
index = load_faiss_index("data/faiss_index/faiss_index.idx")
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

def query_huggingface(prompt):
    """
    Query the Hugging Face Inference API.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        # The output is typically in the format [{"generated_text": "..."}]
        result = response.json()
        return result[0]["generated_text"]
    except requests.RequestException as e:
        logging.error(f"Error querying Hugging Face API: {e}")
        return "The model service is currently unavailable. Please try again later."

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
    Unified API to search for relevant courses and provide a conversational response using the LLM.
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
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)  # Retrieve top 5 for validation

        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Filter results by distance threshold
        threshold = 1.1
        valid_results = []
        query_keywords = set(processed_query.split())

        for i, idx in enumerate(indices[0]):
            if D[0][i] < threshold:
                course = records[idx]
                combined_text = combine_course_fields_with_weights(course).lower()
                if any(keyword in combined_text for keyword in query_keywords):
                    valid_results.append((course, D[0][i]))

        if valid_results:
            top_result, top_distance = valid_results[0]
            platform_info = "an online course" if top_result[11] == "O" else "taught in a university"

            # Query Hugging Face for a conversational response
            response = query_huggingface(
                f"If you're interested in '{query}', you might enjoy the course '{top_result[0]}'.\n\n"
                f"This course is {platform_info}.\n\n"
                f"Here's a quick summary: {top_result[2]}\n\nPlease summarize this course in a conversational and engaging manner."
            )

            return jsonify({
                "result": {
                    "title": top_result[0],
                    "instructor": top_result[1],
                    "learning_obj": top_result[2],
                    "course_contents": top_result[3],
                    "prerequisites": top_result[4],
                    "credits": top_result[5],
                    "evaluation": top_result[6],
                    "time": top_result[7],
                    "frequency": top_result[8],
                    "duration": top_result[9],
                    "course_type": top_result[10],
                    "platform": platform_info
                },
                "response": response
            }), 200

        response = query_huggingface(
            f"No relevant courses were found for the query '{query}'.\n\n"
            "Please summarize why no relevant courses were found and suggest refining the search criteria."
        )
        return jsonify({
            "result": {},
            "response": response
        }), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
