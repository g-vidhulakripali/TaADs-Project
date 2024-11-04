import os
from api_utils import get_learning_obj_en, store_in_faiss

def main():
    # Paths for the database and Faiss index
    db_file = "../data/db/courses.sqlite"
    faiss_index_file = "../data/faiss_index/faiss_index.idx"

    # Get learning objectives from the SQLite database
    records = get_learning_obj_en(db_file)
    if records:
        # Create and store the Faiss index
        store_in_faiss(records, faiss_index_file)
    else:
        print("No records found in the database.")

if __name__ == "__main__":
    main()