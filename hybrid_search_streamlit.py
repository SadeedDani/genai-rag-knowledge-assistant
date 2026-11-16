
# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss

# --IMP-- Reuse components from our existing project files
from gen_ai_rag.doc_loader import section_chunks
from gen_ai_rag.embedding_builder import client, index, embeddings_vectors


# Enable debugging messages in the terminal
DEBUG = True
# DEBUG = False


# ============================================================
# STEP 2: LOAD THE CROSS-ENCODER RERANKER
# ============================================================

# CrossEncoder compares the question and retrieved chunks
# together to calculate more accurate relevance scores.

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# STEP 3: BUILD THE BM25 KEYWORD-SEARCH INDEX
# ============================================================

# Convert every document chunk into lowercase words.
tokenized_chunks = [
    chunk.lower().split()
    for chunk in section_chunks
]

# BM25 uses these tokens for keyword-based retrieval.
bm25 = BM25Okapi(tokenized_chunks)

# --IMP-- Extract the section title from a retrieved chunk
def get_section_title(chunk):
    # The first line contains the section number and title.
    return chunk.strip().split("\n")[0]

if DEBUG:
    print("BM25 ready:", len(section_chunks), "chunks", flush=True)
    print("FAISS ready:", index.ntotal, "vectors", flush=True)


# ============================================================
# STEP 4: CREATE A REUSABLE QUESTION-ANSWERING FUNCTION
# ============================================================

def ask_question(question):

    print("DEBUG 1: Question received", flush=True)

    # Remove unnecessary spaces from the question.
    question = question.strip()

    # Handle an empty question.
    if not question:
        return {
            "answer": "Please enter a question.",
            "context": "",
            "sources": [],
            "debug": {}
        }


    # ========================================================
    # STEP 5: BM25 KEYWORD SEARCH
    # ========================================================

    # Convert the user's question into lowercase words.
    tokenized_question = question.lower().split()

    # Compare the question against every document chunk.
    bm25_scores = bm25.get_scores(tokenized_question)

    # Retrieve the top 3 most relevant chunk indices.
    bm25_indices = np.argsort(bm25_scores)[::-1][:3]


    # ========================================================
    # STEP 6: GENERATE QUESTION EMBEDDING
    # ========================================================

    print("DEBUG 2: Generating question embedding...", flush=True)

    # Gemini converts the question into a numerical vector.
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    print("DEBUG 3: Question embedding generated", flush=True)

    question_embedding = response.embeddings[0].values

    # Convert the embedding into a NumPy array.
    # FAISS expects float32 vectors.
    question_vector = np.array(
        [question_embedding],
        dtype="float32"
    )

    # Normalize the question vector before FAISS search.
    faiss.normalize_L2(question_vector)


    # ========================================================
    # STEP 7: FAISS SEMANTIC SEARCH
    # ========================================================

    print("DEBUG 4: Starting FAISS search", flush=True)

    # Search for the 3 most similar document vectors.
    k = min(3, index.ntotal)

    faiss_scores, faiss_results = index.search(
        question_vector, k
    )

    # Convert FAISS results into a normal Python list.
    faiss_indices = faiss_results[0].tolist()

    print("DEBUG 5: FAISS search completed", flush=True)


    # ========================================================
    # STEP 8: COMBINE BM25 AND FAISS RESULTS
    # ========================================================

    # Remove duplicate indices while preserving their order.
    combined_indices = list(dict.fromkeys(
        bm25_indices.tolist() + faiss_indices
    ))

    # Remove invalid indices such as FAISS's -1 placeholder.
    combined_indices = [
        idx for idx in combined_indices
        if 0 <= idx < len(section_chunks)
    ]


    # ========================================================
    # STEP 9: RETRIEVE ACTUAL TEXT OF CANDIDATE CHUNKS
    # ========================================================

    # Use the retrieved indices to get the actual document text.
    candidate_chunks = [
        section_chunks[idx]
        for idx in combined_indices
    ]


    # ========================================================
    # STEP 10: CROSS-ENCODER RERANKING
    # ========================================================

    # Pair the question with every retrieved candidate chunk.
    pairs = [
        [question, chunk]
        for chunk in candidate_chunks
    ]

    print("DEBUG 6: Starting CrossEncoder reranking", flush=True)

    # Calculate relevance scores for each question-chunk pair.
    rerank_scores = reranker.predict(pairs)

    print("DEBUG 7: CrossEncoder completed", flush=True)

    # Combine scores and chunks, then sort by score.
    ranked = sorted(
        zip(rerank_scores, candidate_chunks),
        key=lambda item: item[0],
        reverse=True
    )


    # ========================================================
    # STEP 11: SELECT TOP 3 CHUNKS
    # ========================================================

    # Keep the top 3 chunks with positive relevance scores.
    top_chunks = [
        chunk for score, chunk in ranked[:3]
        if score > 0
    ]

    # --IMP-- Fallback: if no score is positive,
    # use the highest-ranked chunk rather than empty context.
    if not top_chunks and ranked:
        top_chunks = [ranked[0][1]]

    # Combine all selected chunks into one context string.
    context = "\n\n".join(top_chunks)


    # ========================================================
    # STEP 12: PREPARE THE GEMINI PROMPT
    # ========================================================

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question: {question}

    Mention the relevant section number and title in your answer.
    If the answer is not in the context, say you don't know.
    """


    # ========================================================
    # STEP 13: GENERATE FINAL ANSWER USING GEMINI
    # ========================================================

    print("DEBUG 8: Sending prompt to Gemini...", flush=True)

    try:
        answer_response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )

        print("DEBUG 9: Gemini response received", flush=True)

        answer = answer_response.text or "No answer was generated."

    except Exception as e:
        # Display the actual error instead of hiding it.
        answer = f"Gemini API error: {e}"


    # ========================================================
    # STEP 14: PREPARE DEBUG INFORMATION
    # ========================================================

    debug_info = {}

    if DEBUG:
        debug_info = {
            "BM25 indices": bm25_indices.tolist(),
            "FAISS indices": faiss_indices,
            "Combined indices": combined_indices,

            "Reranker scores": [
                {
                    "score": float(score),
                    "chunk": chunk
                }
                for score, chunk in ranked
            ]
        }


    # ========================================================
    # STEP 15: RETURN RESULTS TO STREAMLIT
    # ========================================================

    # Return a dictionary containing everything app.py needs.
    return {
    "answer": answer,
    "context": context,
    "sources": top_chunks,
    "source_scores": [
        float(score)
        for score, chunk in ranked
        if chunk in top_chunks
    ],
    "debug": debug_info
}

# ============================================================
# END OF RAG PIPELINE
# ============================================================
