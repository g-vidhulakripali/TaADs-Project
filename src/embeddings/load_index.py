import faiss

def load_faiss_index(faiss_index_file):
    try:
        index = faiss.read_index(faiss_index_file)
        return index
    except Exception as e:
        print(f"Error loading Faiss index: {e}")