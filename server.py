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
from sentence_transformers import util

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

@app.route('/search/', methods=['POST'])
async def search():
    """
    Search for relevant courses by matching the query against weighted embeddings of title, learning_obj,
    course_contents, and prerequisites.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid JSON input'}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            return jsonify({"results": "Query is required"}), 400

        # Vectorize the query
        user_vector = cached_vectorize_input(query)

        # Perform FAISS search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)

        # Debugging: Log distances and indices
        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Filter results with adjusted threshold
        matching_results = []
        for idx, dist in zip(indices[0], D[0]):
            if 0 <= idx < len(records) and dist < 1.5:  # Relaxed threshold for flexibility
                record = records[idx]
                matching_results.append({
                    "title": record[0],
                    "instructor": record[1],
                    "learning_obj": record[2],
                    "course_contents": record[3],
                    "prerequisites": record[4],
                    "credits": record[5],
                    "evaluation": record[6],
                    "time": record[7],
                    "frequency": record[8],
                    "duration": record[9],
                    "course_type": record[10]
                })

        # Limit to top 3 results
        matching_results = matching_results[:3]

        if not matching_results:
            return jsonify({"results": "No relevant courses found."}), 200

        return jsonify({"results": matching_results}), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/suggest_course/', methods=['POST'])
async def suggest_course():
    """
    Suggest relevant courses and provide a generative response for question-like queries using Ollama.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Preprocess the query: Remove filler words and non-relevant terms
        def preprocess_query(query):
            query = re.sub(r"[^\w\s]", "", query)  # Remove punctuation
            query = re.sub(r"\b(I|want|like|need|find|please|help|courses|on)\b", "", query, flags=re.IGNORECASE)
            return query.strip().lower()

        processed_query = preprocess_query(query)
        logging.debug(f"Processed query: {processed_query}")

        # Vectorize the processed query
        user_vector = cached_vectorize_input(processed_query)

        # Perform FAISS search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)

        # Log debugging info
        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Filter relevant courses with a relaxed threshold
        threshold = 1.5
        relevant_courses = []
        for idx, dist in zip(indices[0], D[0]):
            if 0 <= idx < len(records) and dist < threshold:
                record = records[idx]
                # Allow loosely matched results based on title and description relevance
                if processed_query in record[0].lower() or processed_query in record[2].lower():
                    relevant_courses.append(record)

        # If no relevant courses are found
        if not relevant_courses:
            return jsonify({
                "response": "No relevant courses found for your query."
            }), 200

        # Format relevant courses into a summary for Ollama
        course_list = "\n".join([f"- {course[0]}: {course[2]}" for course in relevant_courses[:3]])
        prompt = (
            f"The user is searching for courses related to '{query}'. Here are some relevant results:\n\n"
            f"{course_list}\n\n"
            f"Generate a concise, helpful response summarizing these courses in a conversational tone."
        )

        # Generate response using Ollama
        llm_response = ollama_llm(prompt)
        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.error(f"Error in /suggest_course/: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
