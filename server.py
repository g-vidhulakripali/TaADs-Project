import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from util.api_utils import get_learning_obj_en, load_faiss_index, store_in_faiss
from cachetools import cached, TTLCache
from sentence_transformers import SentenceTransformer
import subprocess
import socket
import asyncio
import time

# Configuration
logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME')
db_file = "data/db/courses.sqlite"
faiss_index_file = "data/faiss_index/faiss_index.idx"
OLLAMA_PORT = 11434  # Default port for ollama serve

# Load model once into memory
logging.info("Loading model...")
model = SentenceTransformer(llm_name)

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)

# Preload records and FAISS index
logging.info("Loading records and FAISS index...")
records = get_learning_obj_en(db_file)
if records:
    store_in_faiss(records, faiss_index_file)
else:
    raise Exception("No records found in the database.")

index = load_faiss_index(faiss_index_file)
if index is None or not records:
    raise Exception("Faiss index or database records could not be loaded successfully.")

if index.ntotal != len(records):
    raise Exception(f"FAISS index size ({index.ntotal}) does not match the number of records ({len(records)}).")


def is_port_in_use(port):
    """Check if a specific port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def start_ollama_serve():
    """Start the Ollama server if not already running."""
    if is_port_in_use(OLLAMA_PORT):
        logging.info(f"Ollama server is already running on port {OLLAMA_PORT}.")
    else:
        logging.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # Give some time for the server to start
        if is_port_in_use(OLLAMA_PORT):
            logging.info("Ollama server started successfully.")
        else:
            logging.error("Failed to start Ollama server. Please check your setup.")


# Ensure Ollama server is running
start_ollama_serve()


@cached(vector_cache)
def cached_vectorize_input(user_input):
    """Cache vectorization to speed up repeated queries."""
    start_time = time.time()
    embedding = model.encode([user_input])[0]
    logging.debug(f"Vectorization took {time.time() - start_time:.4f} seconds")
    return embedding


async def run_ollama(prompt, timeout=60):
    """Asynchronous execution of Ollama CLI for response generation with a configurable timeout."""
    logging.info(f"Running Ollama with prompt: {prompt[:100]}...")
    start_time = time.time()
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama", "run", "mistral", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            logging.debug(f"Ollama took {time.time() - start_time:.4f} seconds")
            return stdout.decode().strip(), stderr.decode().strip()
        except asyncio.TimeoutError:
            process.kill()
            logging.error("Ollama CLI timed out.")
            return None, "Timeout: Ollama CLI took too long to respond."
    except Exception as e:
        logging.error(f"Ollama execution failed: {e}")
        return None, str(e)


@app.route('/search/', methods=['POST'])
async def search():
    """Search for relevant courses with optional filters."""
    try:
        logging.info("Received /search/ request")
        start_time = time.time()

        if not request.is_json:
            return jsonify({'error': 'Invalid JSON input'}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "")

        if not query:
            return jsonify({"results": "Query is required"}), 400

        # Vectorize user input
        vector_start = time.time()
        user_vector = cached_vectorize_input(query)
        logging.debug(f"Vectorization took {time.time() - vector_start:.4f} seconds")

        # Perform FAISS search
        search_start = time.time()
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)
        logging.debug(f"FAISS search took {time.time() - search_start:.4f} seconds")

        # Filter and format results
        matching_results = []
        for idx in indices[0]:
            if 0 <= idx < len(records):
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

        logging.info(f"/search/ completed in {time.time() - start_time:.4f} seconds")
        return jsonify({"results": matching_results or "No matching results found."}), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/suggest_course/', methods=['POST'])
async def suggest_course():
    """Suggest relevant courses in a generative way based on the user's query."""
    try:
        logging.info("Received /suggest_course/ request")
        start_time = time.time()

        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Vectorize user input
        vector_start = time.time()
        user_vector = cached_vectorize_input(query)
        logging.debug(f"Vectorization took {time.time() - vector_start:.4f} seconds")

        # Perform FAISS search
        search_start = time.time()
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)
        logging.debug(f"FAISS search took {time.time() - search_start:.4f} seconds")

        # Format prompt for generative response
        relevant_courses = [records[idx] for idx in indices[0] if 0 <= idx < len(records)]
        if relevant_courses:
            course_list = "\n".join([f"- {course[0]}" for course in relevant_courses[:3]])  # Limit to top 3
            prompt = f"The user is looking for courses related to '{query}'. Relevant courses:\n{course_list}"
        else:
            prompt = f"No relevant courses found for the query: '{query}'."

        # Generate response
        ollama_start = time.time()
        response, error = await run_ollama(prompt, timeout=60)  # Increased timeout
        logging.debug(f"Ollama generation took {time.time() - ollama_start:.4f} seconds")

        if response:
            logging.info(f"/suggest_course/ completed in {time.time() - start_time:.4f} seconds")
            return jsonify({"response": response}), 200
        else:
            return jsonify({"error": error}), 500

    except Exception as e:
        logging.error(f"Error in /suggest_course/: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
