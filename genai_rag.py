import os
import re
import json
import chromadb
import google.generativeai as genai

#Load your environment variables if needed
from dotenv import load_dotenv
load_dotenv()

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

def chunk_text(text, max_chunk_size=1000):
    """Chunk the text into smaller pieces."""
    chunks = []
    current_chunk = ''
    
    for line in text.splitlines():
        if len(current_chunk) + len(line) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        current_chunk += line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_json_file(json_file_path):
    """Process the JSON file and upsert data into ChromaDB."""
    try:
        with open(json_file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        for item in data:
            path = item['path']
            content = item['content']
            chunks = chunk_text(content)
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
    You are a highly knowledgeable AI assistant. Your task is to provide a concise response to the user's question based on the given context.
    
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

def main():
    json_file_path = 'output.json'  # Path to your output JSON file
    process_json_file(json_file_path)

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = query_and_generate(query)
        print("Assistant:", answer)

if __name__ == "__main__":
    main()
