import logging
import configparser
import numpy as np
from quart_cors import cors
from quart import Quart, request, jsonify
from pyfiglet import Figlet
from util.api_utils import vectorize_input, get_learning_obj_en, load_faiss_index

config = configparser.ConfigParser()
config.read('config.env')

app = Quart(__name__)
app = cors(app, allow_origin="*")

llm_name = config['DEFAULT'].get('LLM_NAME')
db_courses = config['DEFAULT'].get('DB_COURSES')
faiss_index_file = "./data/faiss_index/faiss_index.idx"

records = get_learning_obj_en(db_courses)
index = load_faiss_index(faiss_index_file)
if index is None or not records:
    raise Exception("Faiss index or database records could not be loaded successfully.")

@app.route('/search/', methods=['POST'])
async def search():
    """
    {
        "query": "Your search query here",
        "instructor_name": "Optional filter",
        "course_type": "Optional filter",
        "duration": "Optional filter",
        "time": "Optional filter",
        "credits": "Optional filter"
    }
    @return: a JSON file with the matching search results
    """
    if request.is_json:
        data_json = await request.get_json()
        query = data_json.get("query")
        instructor_name = data_json.get("instructor_name")
        course_type = data_json.get("course_type")
        duration = data_json.get("duration")
        time = data_json.get("time")
        credits = data_json.get("credits")

        # Vectorize user input
        user_vector = vectorize_input(llm_name, query)

        # Perform search with optional filters
        D, indices = index.search(np.array([user_vector]).astype('float32'), k=3)  # Changed 'top_k' to 'k'
        filtered_indices = []

        for idx in indices[0]:
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
            filtered_indices.append(record)

        if not filtered_indices:
            resp = jsonify({"results": "No matching results found."})
        else:
            matching_results = []
            for record in filtered_indices:
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
            resp = jsonify({"results": matching_results})
    else:
        resp = jsonify({'error': 'Well-formed JSON is required, please check request'})
    return resp

# Run the app using Hypercorn (not recommended for production)
app.run(host="0.0.0.0", debug=False, port=3000)
