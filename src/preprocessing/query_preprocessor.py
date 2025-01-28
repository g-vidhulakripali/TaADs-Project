import re
import difflib

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

# import requests
# import logging
#
# def query_huggingface(prompt, api_url, api_token):
#     """
#     Query the Hugging Face Inference API for text generation.
#
#     Args:
#         prompt (str): The input prompt for the model.
#         api_url (str): The Hugging Face model API endpoint.
#         api_token (str): The API token for authentication.
#
#     Returns:
#         str: The generated text response from the model.
#     """
#     headers = {"Authorization": f"Bearer {api_token}"}
#     payload = {"inputs": prompt}
#
#     try:
#         logging.info(f"Querying Hugging Face API with prompt: {prompt}")
#         response = requests.post(api_url, headers=headers, json=payload)
#         response.raise_for_status()
#         result = response.json()
#
#         # Extract the generated text
#         if isinstance(result, list) and "generated_text" in result[0]:
#             return result[0]["generated_text"].strip()
#         else:
#             logging.error(f"Unexpected response format: {result}")
#             return "The model response could not be processed."
#     except requests.RequestException as e:
#         logging.error(f"Error querying Hugging Face API: {e}")
#         return "The model service is currently unavailable. Please try again later."
#
