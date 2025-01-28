from cachetools import cached, TTLCache
import logging
from sentence_transformers import SentenceTransformer
import re
import difflib
from langchain.llms import Ollama

# Cache for vectorization
vector_cache = TTLCache(maxsize=100, ttl=300)
llm_name = 'all-mpnet-base-v2'
# Load SentenceTransformer model for vectorization
logging.info("Loading SentenceTransformer model for vectorization...")
vector_model = SentenceTransformer(llm_name)

ollama_llm = Ollama(model="mistral")

# Function to preprocess query
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

# Function to extract keywords with LLM
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

# Function to build vocabulary
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

# Vectorization with caching
@cached(vector_cache)
def cached_vectorize_input(user_input):
    """Cache vectorization to speed up repeated queries."""
    return vector_model.encode([user_input])[0]
