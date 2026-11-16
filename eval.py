import faiss
import numpy as np
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import re
import faiss


api_key=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=api_key)


#read file
with open("docs/company_policy.txt", "r") as file:
    text = file.read()

#create chunks
matches = re.split(r"^(\d+\.)", text, flags=re.MULTILINE)
section_chunks = []
for i in range(1, len(matches), 2):
    combine = matches[i] + matches[i+1]
    section_chunks.append(combine)
#create embedding
embeddings = []
for chunk in section_chunks:
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk
    )
    embeddings.append(response.embeddings[0].values)
    




dimension = len(embeddings[0])
embeddings_vectors = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings_vectors)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_vectors)


# question = "What is the notice period?"
# ground_truth = 6
# # normalize...

# ques_response = client.models.embed_content(
#     model="gemini-embedding-2",
#     contents=question,
# )

# question_embedding = ques_response.embeddings[0].values

# question_embedding_vector = np.array(
#     [question_embedding]
# ).astype("float32")
# faiss.normalize_L2(question_embedding_vector)
# scores, indices = index.search(question_embedding_vector, 3)
# print(scores,indices)

# #hit@k
# if ground_truth in indices[0]:
#     print(1)
# else:
#     print(0)

# Hit@3 across multiple questions
# For one question:
# correct retrieved → 1
# not retrieved      → 0

eval_data = [
    ("What is the notice period?", 6),
    ("How many sick leave days are employees entitled to?", 1),
    ("What are the working hours?", 0),
]
#ground_truth = chunk 6 is the correct chunk for this question.
hits=0
mrr_total = 0
for question,ground_truth  in eval_data:
    # print(question)
    # print(ground_truth)

    ques_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question,
    )

    question_embedding = ques_response.embeddings[0].values

    question_embedding_vector = np.array(
        [question_embedding]
    ).astype("float32")
    #Normalize = convert something into a common/standard scale or form.
    faiss.normalize_L2(question_embedding_vector)
    
    scores, indices = index.search(question_embedding_vector, 3)
    
    if ground_truth in indices:
        hits+=1
    #     print(hits)
    # print(scores,indices)



# MRR
# What we're doing: Measuring how high the correct chunk appears in our Top-K results.
# Why: Hit@3 only tells us whether we found it; MRR tells us how highly we found it.
#np.where finds where exaclty position of the correct chunk
# 0][0] extracts that position, and +1 converts Python's 0-based position into a human ranking. 
    rank = np.where(indices[0] == ground_truth)[0][0] + 1
    reciprocal_rank = 1 / rank

    if ground_truth in indices[0]:
        rank = np.where(indices[0] == ground_truth)[0][0] + 1
        reciprocal_rank = 1 / rank
    else:
        reciprocal_rank = 0

    mrr_total += reciprocal_rank
#for hits
# len(eval_data)
# hits_at_3=hits/len(eval_data)
# print("Hit@3:", hits_at_3)

mrr = mrr_total / len(eval_data)
print("MRR:", mrr)
print("Rank:", rank)
print("Reciprocal Rank:", reciprocal_rank)

### Precision@K — Short Notes

# **What:** Measures how much of the retrieved results are actually relevant.

# **Formula:**

# ```text
# Precision@K = Relevant retrieved chunks / Total retrieved chunks
# ```

# ### Example

# ```text
# Top 3 retrieved → [6, 2, 0]
# Relevant chunks → [6]
# ```

# Only **1 of 3** is relevant:

# ```text
# Precision@3 = 1/3 = 0.33 = 33%
# ```

# ### Difference from Hit@3

# ```text
# Hit@3       → Did we find at least one relevant chunk?
# Precision@3 → How many of our retrieved chunks were relevant?
# ```

# ### Code

# ```python
# relevant_retrieved = 0

# for idx in indices[0]:
#     if idx in relevant_chunks:
#         relevant_retrieved += 1

# precision = relevant_retrieved / len(indices[0])
# ```

# **Key idea:**  
# > Hit asks **"Did I find it?"**  
# > Precision asks **"How much of what I found was useful?"**


# ### Implement Recall@3

# **What we're doing:** Counting how many of the known relevant chunks appear in our Top 3.

# **Why:** We want to know whether we're missing relevant information.

# For each question:

# ```python
# relevant_retrieved = 0

# for idx in indices[0]:
#     if idx in relevant_chunks:
#         relevant_retrieved += 1

# recall = relevant_retrieved / len(relevant_chunks)
# ```

# Example:

# ```text
# Relevant:  [1, 2, 3]
# Retrieved: [1, 2, 5]

# relevant_retrieved = 2
# len(relevant_chunks) = 3

# Recall = 2/3 = 0.67
# ```

# ### One important thing

# We can actually calculate **Precision and Recall using the same `relevant_retrieved` count**:

# ```text
# Precision = relevant_retrieved / retrieved_count
# Recall    = relevant_retrieved / relevant_count
# ```

# So you don't need two separate retrieval loops.

# Next, we'll put both into your evaluation code and then **finish retrieval evaluation**.


"For RAG retrieval evaluation, "
"I use Hit@K to check whether relevant chunks appear in the top K,"
" MRR to measure their ranking position, "
"Precision@K to measure retrieval quality,"
" and Recall@K to measure retrieval coverage."