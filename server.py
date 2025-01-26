import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from api_utils import get_learning_obj_en, load_faiss_index, store_in_faiss
from cachetools import cached, TTLCache
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import Ollama
import re
import difflib
import json

# Configuration
logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME', 'all-mpnet-base-v2')  # Updated model for better embeddings
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

    # Replace None with empty strings to prevent concatenation errors
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

def extract_keywords_with_llm(query):
    """
    Use the LLM to extract keywords or refine the intent of the query.
    """
    try:
        refined_query = ollama_llm(f"Extract the main keywords or intent from the following query: '{query}'")
        logging.debug(f"Refined query from LLM: {refined_query}")
        return refined_query.strip()
    except Exception as e:
        logging.error(f"Error extracting keywords with LLM: {e}")
        return query  # Fallback to original query

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

        # Extract main keywords or refine the query
        refined_query = extract_keywords_with_llm(query)

        # Preprocess the query
        processed_query = preprocess_query(refined_query, vocabulary)
        logging.debug(f"Processed query: {processed_query}")

        # Vectorize the processed query
        user_vector = cached_vectorize_input(processed_query)

        # Perform FAISS search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)  # Retrieve top 5 for validation

        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Filter results by distance threshold
        threshold = 1.0  # Stricter threshold for semantic matches
        valid_results = []
        query_keywords = set(processed_query.split())

        for i, idx in enumerate(indices[0]):
            if D[0][i] < threshold:
                course = records[idx]
                combined_text = combine_course_fields_with_weights(course).lower()
                # Check if at least one query keyword is in the course content
                if any(keyword in combined_text for keyword in query_keywords):
                    valid_results.append((course, D[0][i]))

        if valid_results:
            # Select the most relevant result
            top_result, top_distance = valid_results[0]

            # Determine platform and include in response
            platform_info = "an online course" if top_result[11] == "O" else "taught in a university"

            response = ollama_llm(
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
                    "platform": top_result[11],
                    "platform_info": platform_info
                },
                "response": response
            }), 200

        # No relevant course found
        response = ollama_llm(
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
    app.run(host="0.0.0.0", debug=False, port=3000)
