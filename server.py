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

# Initialize the Ollama Mistral model using LangChain
ollama_llm = Ollama(model="mistral")

@app.route('/suggest_course/', methods=['POST'])
async def suggest_course():
    """
    Suggest relevant courses or indicate no relevant results if none are sufficiently related.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Ensure records exist
        if not records:
            return jsonify({"error": "No course records available for suggestions."}), 500

        # Vectorize the query
        user_vector = cached_vectorize_input(query)

        # Perform FAISS search, limiting results to top 5
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)

        # Log distances and indices for debugging
        logging.debug(f"FAISS search distances: {D}")
        logging.debug(f"FAISS search indices: {indices}")

        # Filter relevant courses based on stricter threshold
        threshold = 1.2  # Adjust threshold for stricter relevance
        relevant_courses = [
            records[idx] for idx, dist in zip(indices[0], D[0]) if 0 <= idx < len(records) and dist < threshold
        ]

        # If no relevant courses, return a direct response
        if not relevant_courses:
            return jsonify({
                "response": "No relevant courses found for your query."
            }), 200

        # Format top 3 courses into a summary for response
        summarized_courses = relevant_courses[:3]
        course_list = "\n".join([f"- {course[0]}: {course[2][:50]}..." for course in summarized_courses])

        # Use Ollama to generate a natural response for relevant courses
        prompt = (
            f"The user is searching for courses related to '{query}'. Here are some relevant results:\n\n"
            f"{course_list}\n\n"
            f"Generate a concise and helpful response summarizing these courses in a conversational tone."
        )
        llm_response = ollama_llm(prompt)
        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.error(f"Error in /suggest_course/: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
