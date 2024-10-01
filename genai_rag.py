import os
import json
import chromadb
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

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

   # Modified prompt to include file type context
   prompt = f"""
   You are an AI assistant knowledgeable about various programming languages and configurations.
   The following context includes code snippets and configuration files that may relate to your question.
   
   Context from knowledge base (including programming language details):
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
