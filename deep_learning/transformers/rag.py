from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# 1. Create Document Embeddings
documents = [
    "Solar panels convert sunlight into electricity using photovoltaic cells.",
    "Blueberries are rich in antioxidants, which improve heart health.",
    "Kubernetes is an open-source container orchestration platform.",
    "The Eiffel Tower is located in Paris, France, and is a famous landmark.",
    "Python is a versatile programming language used for web development and data science."
]

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the documents
print("Generating embeddings...")
doc_embeddings = model.encode(documents)

# 2. Store Embeddings in a FAISS Index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
print("Storing embeddings in FAISS index...")
index.add(np.array(doc_embeddings))

# 3. Query Processing
def retrieve_relevant_docs(query, top_k=2):
    # Generate an embedding for the query
    query_embedding = model.encode(query).reshape(1, -1)
    
    # Search for the top_k nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding documents
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs

# 4. Pass Query and Context to GPT
def generate_response_with_context(query):
    # Retrieve relevant documents
    context = retrieve_relevant_docs(query)
    context_str = "\n".join(context)
    
    # Combine context and query for GPT
    prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
    
    # Call OpenAI GPT API
    print("Generating response from GPT...")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example Query
query = "How do solar panels work?"
response = generate_response_with_context(query)

# Display the Result
print("Query:", query)
print("Response:", response)
