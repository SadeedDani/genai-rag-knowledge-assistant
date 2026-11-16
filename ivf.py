#IVF index — IndexIVFFlat
#Instead of searching all 1 million embeddings, IVF organises them into clusters.

# One important difference
# IndexFlatIP gives exact nearest-neighbour search.

# IndexIVFFlat generally provides approximate nearest-neighbour search because 
# it might skip clusters containing relevant vectors.


# The quantizer helps FAISS find the most suitable clusters for a query.
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist,
    M,
    nbits
)
#FAISS uses K-means clustering to learn the cluster centres (centroids) from your existing embeddings.
index.train(embeddings_vectors)
# FAISS takes the actual document embeddings and assigns each one to its nearest cluster, then stores them in the index.
index.add(embeddings_vectors)
# This tells FAISS to search the 10 nearest clusters for each query.
index.nprobe = 10


#Product Quantization (PQ): reduce memory usage by compressing embeddings.
# One important trade-off
# Compression saves memory, but it can reduce search accuracy 
# because FAISS works with compressed representations rather than the original full vectors.

# Example: Compressing an embedding
# Suppose we have an embedding containing 8 numbers:
# embedding = [0.12, 0.45, 0.78, 0.23,             0.91, 0.34, 0.56, 0.67]



# Step 1: Divide it into 4 subvectors

# Each subvector contains 2 numbers.
# Subvector 1
# 0.12, 0.45

# Subvector 2
# 0.78, 0.23

# Subvector 3
# 0.91, 0.34

# Subvector 4
# 0.56, 0.67



# Step 2: Match each subvector with its codebook

# Imagine each subvector has a codebook containing 4 representative patterns.

# PQ finds the closest representative pattern and stores its ID.

# For our imaginary example:
# Subvector	Closest codebook entry	Stored code
# 1	Entry 2	2
# 2	Entry 0	0
# 3	Entry 3	3
# 4	Entry 1	1

# Step 3: What do we actually store?

# Instead of storing all 8 original numbers, we store:
# compressed_embedding = [2, 0, 3, 1]



# 4 compact codes
# That's the basic idea of Product Quantization!

# Note: These numbers are illustrative, not actual FAISS-generated codebook results.


dimension = len(embeddings[0])
