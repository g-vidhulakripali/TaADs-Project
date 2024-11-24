import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from util.api_utils import vectorize_input, get_learning_obj_en, load_faiss_index, store_in_faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load configuration
config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

# Load configurations
llm_name = config['DEFAULT'].get('LLM_NAME')
db_courses = config['DEFAULT'].get('DB_COURSES')
db_file = "data/db/courses.sqlite"
faiss_index_file = "data/faiss_index/faiss_index.idx"

# Load database and FAISS index
records = get_learning_obj_en(db_file)
if records:
    store_in_faiss(records, faiss_index_file)
else:
    print("No records found in the database.")

index = load_faiss_index(faiss_index_file)
if index is None or not records:
    raise Exception("Faiss index or database records could not be loaded successfully.")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Add a padding token if not defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Set pad_token_id to eos_token_id (50256 for GPT-2)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Initialize text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

print(f"Pad Token ID: {model.config.pad_token_id}")
print(f"EOS Token ID: {model.config.eos_token_id}")


@app.route('/search_and_generate/', methods=['POST'])
async def search_and_generate():
    """
    Unified API to search for relevant courses and generate a targeted response.
    {
        "query": "Your search query"
    }
    """
    if request.is_json:
        data_json = await request.get_json()
        query = data_json.get("query")

        if not query:
            return jsonify({'error': 'Query text is required'})

        # Step 1: Vectorize user input
        user_vector = vectorize_input(llm_name, query)

        # Step 2: Perform search in FAISS index
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)
        results = [records[idx] for idx in indices[0] if idx < len(records)]

        if not results:
            return jsonify({"results": "No matching results found."})

        # Step 3: Format results for grounding
        matching_results = [{
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
        } for record in results]

        # Step 4: Create grounding text from retrieved data
        grounding_text = ". ".join([f"{result['title']}: {result['learning_obj']}" for result in matching_results])

        # Step 5: Generate concise response focused on query and course data
        prompt = f"The university offers the following courses: {grounding_text}. Based on these courses, answer this: {query}"
        generated_text = generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True)[0]["generated_text"]

        # Step 6: Return both retrieved courses and concise response
        return jsonify({"courses": matching_results, "response": generated_text})
    else:
        return jsonify({'error': 'Well-formed JSON is required, please check request'})

# Run the app
app.run(host="0.0.0.0", debug=False, port=3000)
