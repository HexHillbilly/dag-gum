import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Configuration
CORPUS_FILE = "brain_corpus.txt"
INDEX_FILE = "brain.index"
MAP_FILE = "brain_lines.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def compile_brain():
    print(f"[*] Booting up the embedding model: {MODEL_NAME}...")
    # This will download the ~80MB model on the first run, then cache it locally.
    model = SentenceTransformer(MODEL_NAME)

    print(f"[*] Reading raw text from {CORPUS_FILE}...")
    if not os.path.exists(CORPUS_FILE):
        print(f"[!] Error: Could not find {CORPUS_FILE}. Create it first.")
        return

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        # Strip whitespace and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("[!] Corpus is empty.")
        return

    print(f"[*] Translating {len(lines)} lines into vectors. This might take a second...")
    # Convert text to mathematical embeddings
    embeddings = model.encode(lines, convert_to_numpy=True)

    # FAISS expects float32
    embeddings = np.array(embeddings).astype("float32")

    print("[*] Building the FAISS index...")
    # Dimension of the MiniLM embeddings is 384
    dimension = embeddings.shape[1] 
    
    # Using L2 distance (Euclidean) for the index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"[*] Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)

    print(f"[*] Saving text map to {MAP_FILE}...")
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=4)

    print("[+] Brain compiled successfully.")

if __name__ == "__main__":
    compile_brain()
