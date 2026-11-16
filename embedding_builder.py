
# STEP 1: Import required libraries
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
from pathlib import Path
from gen_ai_rag.doc_loader import section_chunks

# --IMP-- Define where the FAISS index is stored
index_path = Path("company_policy.index")
vectors_path = Path("company_policy_vectors.npy")

# STEP 2: Set up the Gemini client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --IMP-- Check whether the saved FAISS index already exists
# --IMP-- Load only when BOTH saved files exist
if index_path.exists() and vectors_path.exists():

    # Load the existing index without regenerating embeddings
    index = faiss.read_index(str(index_path))
    embeddings_vectors = np.load("company_policy_vectors.npy")

    print("Loaded existing FAISS index")

else:
    # STEP 3: Generate an embedding for each chunk
    embeddings = []

    for chunk in section_chunks:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk
        )
        embeddings.append(response.embeddings[0].values)

    # STEP 4: Convert vectors to FAISS format
    embeddings_vectors = np.array(
        embeddings, dtype="float32"
    )

    # STEP 5: Normalize vectors
    faiss.normalize_L2(embeddings_vectors)
    # --IMP-- Save embedding vectors so we don't regenerate them
    np.save("company_policy_vectors.npy", embeddings_vectors)
    # STEP 6: Build the FAISS index
    dimension = embeddings_vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_vectors)

    # --IMP-- Save the newly created index
    faiss.write_index(index, str(index_path))
    print("Created and saved new FAISS index")

# STEP 7: Display index information
if __name__ == "__main__":
    print("Total chunks:", len(section_chunks))
    print("Vectors stored:", index.ntotal)

# Short notes
# - embeddings.append(...) stores one vector per chunk.
# - float32 is the numeric format FAISS expects.
# - Normalization makes each vector length 1.
# - IndexFlatIP uses inner product; with normalized vectors, this gives cosine similarity.
# - index.add() stores the document vectors.
# - --IMP-- The saved index prevents repeated embedding API calls.
