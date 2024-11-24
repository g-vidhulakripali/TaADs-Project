import logging
import configparser
import numpy as np
import torch
from quart_cors import cors
from quart import Quart, request, jsonify
from util.api_utils import vectorize_input, get_learning_obj_en, load_faiss_index, store_in_faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
token = "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"

config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME')
db_courses = config['DEFAULT'].get('DB_COURSES')
db_file = "data/db/courses.sqlite"
faiss_index_file = "data/faiss_index/faiss_index.idx"
records = get_learning_obj_en(db_file)
# Load Mistral model and tokenizer
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=token)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name, token=token)

mistral_model = mistral_model.to("cuda" if torch.cuda.is_available() else "cpu")

if records:
    store_in_faiss(records, faiss_index_file)
else:
    print("No records found in the database.")


index = load_faiss_index(faiss_index_file)
if index is None or not records:
    raise Exception("Faiss index or database records could not be loaded successfully.")

@app.route('/search/', methods=['POST'])
async def search_and_generate():
    """
    {
        "query": "Your search query here"
    }
    @return: JSON response containing the retrieved results and generated summary.
    """
    if not request.is_json:
        return jsonify({'error': 'Well-formed JSON is required, please check request'}), 400

    data_json = await request.get_json()
    query = data_json.get("query", "").strip()

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Step 1: Vectorize the query
    user_vector = vectorize_input(llm_name, query)
    user_vector = user_vector / np.linalg.norm(user_vector)

    # Step 2: Search in FAISS
    D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)  # Top-3 results
    results = []

    for idx in indices[0]:
        record = records[idx]
        match_score = 1 - D[0][indices[0].tolist().index(idx)]  # Cosine similarity score
        results.append({
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
            "course_type": record[10],
            "score": match_score
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)


    if not results:
        return jsonify({"results": "No matching results found."})

    # Step 3: Generate a summary using the GPT model
    context = "\n".join([f"- {res['learning_obj']} by {res['title']}" for res in results])
    prompt = (
    f"The user asked: {query}\n"
    f"Here are the top matching courses with details:\n"
    f"{context}\n\n"
    "Summarize these courses in an engaging and concise manner:"
    )
    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral_model.device)
    outputs = mistral_model.generate(
        inputs["input_ids"],
        max_length=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    generated_summary = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    structured_summary = {
        "summary": generated_summary,
    }

    # Return the results and summary
    return jsonify({"retrieved_results": results, "generated_summary": structured_summary})

# Run the app using Hypercorn (not recommended for production)
try:
    app.run(host="0.0.0.0", port=3000, debug=True)
except Exception as e:
    print(f"Error while running the server: {e}")