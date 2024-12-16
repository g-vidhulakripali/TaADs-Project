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
logging.info("Initializing Ollama for keyword extraction...")
ollama_llm = Ollama(model="mistral")

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)

# Load records from the database
logging.info("Loading records from the database...")
records = get_learning_obj_en(db_file)
if not records:
    raise Exception("No records found in the database.")


# Helper Functions

def combine_course_fields_with_weights(course):
    title_weight = 3
    learning_obj_weight = 2
    course_contents_weight = 1
    prerequisites_weight = 1

    title = (course[0] + " ") * title_weight
    learning_obj = (course[2] + " ") * learning_obj_weight
    course_contents = (course[3] + " ") * course_contents_weight
    prerequisites = (course[4] + " ") * prerequisites_weight

    return f"{title}{learning_obj}{course_contents}{prerequisites}"


def store_weighted_embeddings(records, faiss_index_file):
    weighted_texts = [combine_course_fields_with_weights(record) for record in records]
    embeddings = vector_model.encode(weighted_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_index_file)
    logging.info("FAISS index with weighted embeddings stored successfully.")


store_weighted_embeddings(records, faiss_index_file)

index = load_faiss_index(faiss_index_file)
if index is None or index.ntotal != len(records):
    raise Exception("FAISS index could not be loaded or does not match the number of records.")


@cached(vector_cache)
def cached_vectorize_input(user_input):
    return vector_model.encode([user_input])[0]


def clean_keywords(raw_keywords):
    """
    Clean and format keywords extracted from the LLM to make them usable for vector search.
    """
    clean_keywords = []
    for keyword in raw_keywords:
        # Remove numbers, parentheses, and extra text
        keyword = re.sub(r"\d+\.|\(.*?\)", "", keyword).strip()
        if keyword:
            clean_keywords.append(keyword)
    return clean_keywords


def extract_keywords_with_model(query):
    """
    Use Ollama or MiniLM to extract and clean keywords from the query.
    """
    prompt = (
        f"Extract only the key topics or keywords from the following query: '{query}'. "
        "Return the keywords as a comma-separated list without explanations."
    )
    response = ollama_llm(prompt)
    raw_keywords = response.split(",")
    return clean_keywords([keyword.strip().lower() for keyword in raw_keywords])


def build_vocabulary(records):
    vocabulary = set()
    for record in records:
        for field in record:
            if isinstance(field, str):
                vocabulary.update(field.lower().split())
    return list(vocabulary)


def aggregate_faiss_results(keywords, index, records, top_k=5, threshold=0.8):
    """
    Perform FAISS search for each keyword and aggregate results with a combined score.
    """
    results = {}
    for keyword in keywords:
        keyword_vector = cached_vectorize_input(keyword)
        D, indices = index.search(np.array([keyword_vector]).astype('float32'), k=top_k)

        for dist, idx in zip(D[0], indices[0]):
            if 0 <= idx < len(records) and dist < threshold:
                if idx not in results:
                    results[idx] = {"score": 0, "record": records[idx]}
                results[idx]["score"] += 1 / (1 + dist)  # Higher scores for closer distances

    # Sort by combined score
    sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    return [res["record"] for res in sorted_results]


def rank_courses(query, courses, user_vector, extracted_keywords):
    ranked_courses = []
    for course in courses:
        course_vector = vector_model.encode([combine_course_fields_with_weights(course)])[0]
        semantic_similarity = np.dot(user_vector, course_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(course_vector)
        )

        title_relevance = sum(1 for kw in extracted_keywords if kw in course[0].lower())
        learning_obj_relevance = sum(1 for kw in extracted_keywords if kw in course[2].lower())

        score = semantic_similarity + (2 * title_relevance) + (1.5 * learning_obj_relevance)
        ranked_courses.append((score, course))

    ranked_courses.sort(key=lambda x: x[0], reverse=True)
    return [course for _, course in ranked_courses]


@app.route('/search/', methods=['POST'])
async def search():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON input"}), 400

        data_json = await request.get_json()
        query = data_json.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Extract keywords using the model
        extracted_keywords = extract_keywords_with_model(query)
        logging.debug(f"Extracted keywords: {extracted_keywords}")

        if not extracted_keywords:
            return jsonify({"results": [], "response": "No relevant keywords found. Please refine your query."}), 200

        # Generate a weighted query for vector search
        weighted_query = " ".join(extracted_keywords)
        user_vector = cached_vectorize_input(weighted_query)

        # Aggregate FAISS results using keywords
        relevant_records = aggregate_faiss_results(extracted_keywords, index, records)

        if not relevant_records:
            return jsonify({"results": [], "response": "No relevant courses found."}), 200

        # Rank courses (if multiple records are aggregated)
        ranked_courses = rank_courses(weighted_query, relevant_records, user_vector, extracted_keywords)

        if not ranked_courses:
            return jsonify({"results": [], "response": "No relevant courses found."}), 200

        # Return the most relevant course
        course = ranked_courses[0]
        json_result = {
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
        }
        
        prompt = (
            f"The user is searching for courses related to '{query}'. Here's the most relevant course:\n\n"
            f"- {course[0]}: {course[2]}\n\n"
            f"Generate a concise, helpful response summarizing this course in a conversational tone."
        )

        llm_response = ollama_llm(prompt)

        return jsonify({"result": json_result, "response": llm_response}), 200

    except Exception as e:
        logging.error(f"Error in /search/: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=3000)
