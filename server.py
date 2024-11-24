import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from util.api_utils import get_learning_obj_en, load_faiss_index, store_in_faiss
from cachetools import cached, TTLCache
from sentence_transformers import SentenceTransformer
import asyncio

# Configuration
config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME')
db_file = "data/db/courses.sqlite"
faiss_index_file = "data/faiss_index/faiss_index.idx"

# Load model once into memory
model = SentenceTransformer(llm_name)

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)

# Preload records and FAISS index for faster access
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

@cached(vector_cache)
def cached_vectorize_input(user_input):
    """Cache vectorization to speed up repeated queries."""
    return model.encode([user_input])[0]

async def run_ollama(prompt):
    """Asynchronous execution of Ollama CLI for response generation."""
    process = await asyncio.create_subprocess_exec(
        "ollama", "run", "mistral",  prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode().strip(), stderr.decode().strip()

@app.route('/search/', methods=['POST'])
async def search():
    """Search for relevant courses with optional filters."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid JSON input'}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "")
        instructor_name = data_json.get("instructor_name", "")
        course_type = data_json.get("course_type", "")
        duration = data_json.get("duration", "")
        time = data_json.get("time", "")
        credits = data_json.get("credits", "")

        if not query:
            return jsonify({"results": "Query is required"}), 400

        # Vectorize user input
        user_vector = cached_vectorize_input(query)

        # Perform search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)

        matching_results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(records):
                continue
            record = records[idx]
            if instructor_name and instructor_name not in record[1]:
                continue
            if course_type and course_type not in record[10]:
                continue
            if duration and duration not in record[9]:
                continue
            if time and time not in record[7]:
                continue
            if credits and credits not in record[5]:
                continue
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

        return jsonify({"results": matching_results or "No matching results found."}), 200

    except Exception as e:
        logging.error(f"Error in search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/suggest_course/', methods=['POST'])
async def suggest_course():
    """Suggest relevant courses in a generative way based on the user's query."""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Vectorize user input
        user_vector = cached_vectorize_input(query)

        # Perform FAISS search
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)

        relevant_courses = [records[idx] for idx in indices[0] if 0 <= idx < len(records)]
        if relevant_courses:
            course_list = "\n".join([f"- {course[0]}" for course in relevant_courses])
            prompt = f"The user is looking for courses related to '{query}'. Relevant courses:\n{course_list}"
        else:
            prompt = f"No relevant courses found for the query: '{query}'."

        # Generate response via Ollama
        response, error = await run_ollama(prompt)
        if response:
            return jsonify({"response": response}), 200
        else:
            return jsonify({"error": error}), 500

    except Exception as e:
        logging.error(f"Error in suggest_course: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
