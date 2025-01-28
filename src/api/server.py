import logging
import os
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from src.db.db_utils import get_learning_obj_en
from src.embeddings.embeddings_handler import store_embeddings_and_faiss, load_faiss_index
from src.preprocessing.query_preprocessor import preprocess_query, build_vocabulary, extract_keywords_with_llm, cached_vectorize_input
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = 'all-mpnet-base-v2'
db_file = "../../data/db/courses.sqlite"
faiss_index_file = "../../data/faiss_index/faiss_index.idx"
pkl_file = "../../data/models/all-MiniLM-L6-v2_embeddings_en.pkl"

# Load SentenceTransformer model for vectorization
logging.info("Loading SentenceTransformer model for vectorization...")
vector_model = SentenceTransformer(llm_name)

# Initialize the Ollama model using LangChain
logging.info("Initializing Ollama for generative responses...")
ollama_llm = Ollama(model="mistral")

# Load records from the database
logging.info("Loading records from the database...")
records = get_learning_obj_en(db_file)
if not records:
    logging.error("No records found in the database.")
    raise Exception("No records found in the database.")
else:
    logging.info(f"Successfully loaded {len(records)} records from the database.")

# Check and handle FAISS index
if not os.path.exists(faiss_index_file):
    logging.info(f"FAISS index not found at {faiss_index_file}. Generating embeddings and creating FAISS index...")
    try:
        store_embeddings_and_faiss(records, faiss_index_file, vector_model)
        logging.info("FAISS index successfully created.")
    except Exception as e:
        logging.error(f"Error during FAISS index creation: {e}")
        raise e
else:
    logging.info(f"FAISS index found at {faiss_index_file}. Attempting to load the index...")
index = load_faiss_index(faiss_index_file)

# Validate the FAISS index
if index is None:
    logging.error(f"Failed to load FAISS index from {faiss_index_file}.")
    raise Exception("FAISS index could not be loaded.")
elif index.ntotal != len(records):
    logging.error(f"FAISS index validation failed: index.ntotal={index.ntotal}, expected={len(records)}.")
    raise Exception("FAISS index does not match the number of records.")
else:
    logging.info(f"FAISS index successfully loaded with {index.ntotal} entries.")

@app.route('/search/', methods=['POST'])
async def search():
    """
    Unified API to search for relevant courses and provide a conversational response using Ollama.
    """
    try:
        if not request.is_json:
            logging.warning("Invalid JSON input received.")
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            logging.warning("Empty query received.")
            return jsonify({"error": "Query is required"}), 400

        # Build a dynamic vocabulary
        logging.info("Building vocabulary from records...")
        vocabulary = build_vocabulary(records)

        # Extract main keywords or refine the query
        logging.info(f"Refining query using LLM: '{query}'")
        refined_query = extract_keywords_with_llm(query)

        # Preprocess the query
        logging.info(f"Preprocessing the query: '{refined_query}'")
        processed_query = preprocess_query(refined_query, vocabulary)
        logging.debug(f"Processed query: {processed_query}")

        # Vectorize the processed query
        logging.info("Vectorizing the processed query...")
        user_vector = cached_vectorize_input(processed_query)

        # Perform FAISS search
        logging.info("Performing FAISS search...")
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=5)

        logging.debug(f"FAISS distances: {D[0]}")
        logging.debug(f"FAISS indices: {indices[0]}")

        # Filter results by distance threshold
        threshold = 1.0
        valid_results = []
        query_keywords = set(processed_query.split())

        for i, idx in enumerate(indices[0]):
            if D[0][i] < threshold:
                course = records[idx]
                valid_results.append(course)

        if valid_results:
            logging.info(f"Found {len(valid_results)} valid results. Preparing the top result...")
            top_result = valid_results[0]
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
        logging.info("No relevant courses found for the query.")
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
