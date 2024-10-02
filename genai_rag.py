import os
import json
import chromadb
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection_name = 'file-embeddings'
collection = chroma_client.get_or_create_collection(name=collection_name)

# Configure Google AI
api_key = os.getenv("GOOGLE_AI_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1000,
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Load a pre-trained sentence transformer model for semantic chunking
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text_semantically(text, max_chunk_size=1000, similarity_threshold=0.7):
    """Chunk the text semantically."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    
    # Generate embeddings for each sentence
    embeddings = semantic_model.encode(sentences)
    
    for i, sentence in enumerate(sentences):
        # If adding this sentence exceeds the max chunk size, save the current chunk
        if sum(len(s) for s in current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append('. '.join(current_chunk).strip())
                current_chunk = []
        
        # Check similarity with the last sentence in the current chunk
        if current_chunk:
            last_embedding = embeddings[len(current_chunk) - 1]
            current_embedding = embeddings[i]
            similarity = np.dot(last_embedding, current_embedding) / (np.linalg.norm(last_embedding) * np.linalg.norm(current_embedding))
            
            if similarity < similarity_threshold:
                # Start a new chunk if the similarity is below the threshold
                chunks.append('. '.join(current_chunk).strip())
                current_chunk = []
        
        # Add the current sentence to the chunk
        current_chunk.append(sentence)

    # Add any remaining sentences as a final chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk).strip())
    
    return chunks

def process_json_file(json_file_path):
    """Process the JSON file and upsert data into ChromaDB."""
    try:
        with open(json_file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        for item in data:
            path = item['path']
            content = item['content']
            chunks = chunk_text_semantically(content)
            ids = [f"{path}_{i}" for i in range(len(chunks))]
            metadatas = [{"file_path": path} for _ in chunks]
            
            collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Processed and upserted content from {path}")
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")

def query_and_generate(query_text):
    """Query the ChromaDB collection and generate a response."""
    results = collection.query(
        query_texts=[query_text],
        n_results=5
    )
    
    distances = results["distances"][0]
    distance_threshold = 1.5

    relevant_docs = [doc for i, doc in enumerate(results["documents"][0]) if distances[i] < distance_threshold]
    
    if not relevant_docs:
        context = "The query does not closely match any specific file content."
    else:
        context = "\n\n".join(relevant_docs)

    prompt = f"""
    You are a highly knowledgeable AI assistant with access to a personal knowledge base containing various programming files. 
    Your task is to provide concise responses based on the context of these files.
    
    Context from knowledge base:
    {context}

    Human: {query_text}

    Assistant:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return str(e)

@app.route('/process', methods=['POST'])
def process_files():
    json_file_path = request.json.get('json_file_path', 'output.json')  # Default to output.json if not provided
    process_json_file(json_file_path)
    return jsonify({"message": "Processing complete."}), 200

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query_text = data.get('query')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    result = query_and_generate(query_text)
    return jsonify({"answer": result}), 200

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()
