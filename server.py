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
    # Define common terms to remove
    filler_words = ["i", "want", "like", "need", "find", "please", "help", "courses", "couse", "course", "on"]

    # Remove punctuation
    query = re.sub(r"[^\w\s]", "", query)

    # Split query into words and process each word
    processed_words = []
    for word in query.lower().split():
        # Remove filler words
        if word in filler_words:
            continue
        # Correct misspellings using fuzzy matching
        matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)  # Match threshold 80%
        if matches:
            processed_words.append(matches[0])
        else:
            processed_words.append(word)

    # Rejoin processed words into a clean query
    return " ".join(processed_words).strip()


def build_vocabulary(records):
    """
    Dynamically build a vocabulary from course titles, descriptions, and other terms in the records.
    """
    vocabulary = set()
    for record in records:
        for field in record:  # Loop through all fields in the record
            if isinstance(field, str):  # Ensure the field is a string
                vocabulary.update(field.lower().split())  # Split and add words to the vocabulary
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
                relevant_courses.append(record)

        # Prepare JSON response
        json_results = []
        for course in relevant_courses[:3]:
            json_results.append({
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
            })

        # If no relevant courses are found
        if not relevant_courses:
            return jsonify({
                "results": [],
                "response": "No relevant courses found for your query."
            }), 200

        # Format relevant courses into a summary for Ollama
        if len(relevant_courses) == 1:
            course_list = f"- {relevant_courses[0][0]}: {relevant_courses[0][2]}"
            prompt = (
                f"The user is searching for courses related to '{query}'. Here's a relevant course:\n\n"
                f"{course_list}\n\n"
                f"Generate a concise, helpful response summarizing this course in a conversational tone."
            )
        else:
            course_list = "\n".join([f"- {course[0]}: {course[2]}" for course in relevant_courses[:3]])
            prompt = (
                f"The user is searching for courses related to '{query}'. Here are some relevant results:\n\n"
                f"{course_list}\n\n"
                f"Generate a concise, helpful response summarizing these courses in a conversational tone."
            )

        # Generate response using Ollama
        llm_response = ollama_llm(prompt)

        # Return both JSON results and conversational response
        return jsonify({
            "results": json_results,
            "response": llm_response
        }), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
