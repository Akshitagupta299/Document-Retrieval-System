import os
import faiss
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document
import spacy
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-pro')

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths for FAISS index
INDEX_FILE = "document_index.faiss"
CHUNKS_FILE = "text_chunks.npy"
DOCUMENTS_FILE = "document_mapping.npy"

# 1️⃣ READ FILES
def read_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def read_docx(file_path):
    """Extract text from a Word file."""
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_txt(file_path):
    """Read text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 2️⃣ SPLIT TEXT INTO CHUNKS
def split_text_into_chunks(text, chunk_size=500):
    """Split text into chunks."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# 3️⃣ PROCESS DOCUMENTS
def preprocess_documents(path):
    """Process documents and split them into chunks."""
    print(f"📂 Processing: {path}")

    text_chunks = []
    document_mapping = []

    if os.path.isfile(path):
        print(f"📄 Single file detected: {path}")
        if path.endswith('.pdf'):
            text = read_pdf(path)
        elif path.endswith('.docx'):
            text = read_docx(path)
        elif path.endswith('.txt'):
            text = read_txt(path)
        else:
            print(f"⚠️ Unsupported format: {path}")
            return [], []

        if not text.strip():
            print(f"❌ No text extracted from {path}")
            return [], []

        chunks = split_text_into_chunks(text)
        text_chunks.extend(chunks)
        document_mapping.extend([os.path.basename(path)] * len(chunks))

    return text_chunks, document_mapping

# 4️⃣ GENERATE EMBEDDINGS & SAVE TO FAISS
def save_faiss_index(chunks, document_mapping):
    """Save embeddings to FAISS index."""
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, chunks, allow_pickle=True)
    np.save(DOCUMENTS_FILE, document_mapping, allow_pickle=True)
    
    print(f"✅ FAISS index saved with {len(chunks)} chunks.")

# 5️⃣ SEARCH FAISS INDEX
def search_query(query, top_k=5):
    """Search for the top_k most relevant chunks."""
    index = faiss.read_index(INDEX_FILE)
    text_chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    document_mapping = np.load(DOCUMENTS_FILE, allow_pickle=True)

    query_embedding = embed_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = [{"text": text_chunks[i], "document": document_mapping[i]} for i in indices[0]]
    return results

# 6️⃣ GENERATE RESPONSE FROM LLM
def generate_llm_response(query, context):
    """Generate AI response using Gemini model."""
    prompt = f"""
    You are a helpful assistant. Answer based on the provided context.
    Context:
    {context}
    Question: {query}
    """
    response = llm_model.generate_content(prompt)
    return response.text

def save_faiss_index(embeddings, chunks, document_mapping):
    """
    Save embeddings into a FAISS index and store the text chunks with their document mappings.
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # Save the FAISS index, chunks, and document mapping
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, chunks)
    np.save(DOCUMENTS_FILE, document_mapping)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
def generate_embeddings(text_chunks):
    """Generate embeddings for text chunks"""
    embeddings = [generate_query_embedding(chunk) for chunk in text_chunks]  # Ensure each embedding is a vector
    
    print(f"🔎 Debug: Embedding shape - {len(embeddings)}x{len(embeddings[0])}")  # Check the shape

    return embeddings

def generate_query_embedding(text):
    """Generate embedding vector from text"""
    embedding = embedding_model.encode([text])  # Ensure it returns a vector
    return embedding.tolist()  # Convert NumPy array to Python list


def search_query(query, top_k=5):
    """
    Search the FAISS index for the top_k most relevant chunks for the user query.
    """
    index = faiss.read_index(INDEX_FILE)
    text_chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    document_mapping = np.load(DOCUMENTS_FILE, allow_pickle=True)

    query_embedding = generate_query_embedding(query).astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [(text_chunks[i], distances[0][j], document_mapping[i]) for j, i in enumerate(indices[0])]
    return results

# Additional Features
def convert_distance_to_similarity(distance, max_distance):
    """Convert L2 distance to a normalized similarity score."""
    return 1 - (distance / max_distance)

def format_retrieved_chunks(results: List[Tuple[str, float, str]]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.
    
    Args:
        results: List of tuples containing (chunk_text, distance, document_name)
    Returns:
        Formatted context string
    """
    context = "Context from documents:\n\n"
    for i, (chunk, distance, doc_name) in enumerate(results, 1):
        context += f"Document: {doc_name}\n"
        context += f"Excerpt {i}:\n{chunk}\n\n"
    return context

def generate_llm_response(query: str, context: str) -> str:
    """
    Generate a response using Gemini based on the query and retrieved context.
    
    Args:
        query: User's question
        context: Retrieved context from documents
    Returns:
        LLM generated response
    """
    try:
        prompt = f"""You are a helpful document analysis assistant. 
        Answer questions based on the provided context. If the answer cannot be 
        found in the context, say so clearly. Always cite specific documents 
        when providing information.
        
        Context:
        {context}
        
        Question: {query}
        
        Please provide a detailed answer based on the context above."""

        response = embedding_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating response: {str(e)}"

def enhanced_display_results(results: List[Tuple[str, float, str]], 
                           max_distance: float, 
                           query: str, 
                           base_path: str):
    """
    Enhanced display function that shows both retrieved chunks and LLM-generated response.
    """
    print("\nAnalyzing retrieved documents...")
    
    # Format context from retrieved chunks
    context = format_retrieved_chunks(results)
    
    # Generate LLM response
    print("\nGenerating detailed response using Gemini...")
    llm_response = generate_llm_response(query, context)
    
    print("\n=== AI Generated Response ===")
    print(llm_response)
    
    # Show individual results with similarity scores
    print("\n=== Retrieved Document Chunks ===")
    for i, (chunk, distance, document) in enumerate(results, 1):
        similarity = convert_distance_to_similarity(distance, max_distance)
        print(f"\nResult {i}:")
        print(f"Document: {document}")
        print(f"Text: {chunk}")
        print(f"Similarity Score: {similarity:.4f}") #  similarity score is a measure of how closely a text chunk from your documents matches the user's query.
        # A higher similarity score indicates that the chunk is more relevant to the query.
        # A lower similarity score suggests that the chunk is less relevant.
        
        show_doc = input("Would you like to view the full document? (yes/no): ").strip().lower()
        if show_doc == "yes":
            document_path = os.path.join(base_path, document)
            if os.path.exists(document_path):
                print(f"\nFull Document: {document}\n")
                with open(document_path, 'r', encoding='utf-8') as doc:
                    print(doc.read())
            else:
                print(f"Error: File not found at {document_path}")

def main():
    """
    Main function with enhanced error handling and user interaction.
    """
    print("Document QA System Initializing...")
    
    # Load environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        print("Error: Google API key not found. Please set GOOGLE_API_KEY in your environment variables.")
        return

    # Get document path
    base_path = input("Enter the path to your document or document folder: ").strip()
    
    if not os.path.exists(base_path):
        print("Error: Invalid path provided.")
        return

    # Process documents
    print("Processing documents...")
    chunks, document_mapping = preprocess_documents(base_path)

    if not chunks:
        print("No valid documents found.")
        return

    # print(f"Processed {len(chunks)} chunks.")
    # print("Generating embeddings...")
    # embeddings = generate_embeddings(chunks)
    # print("Saving FAISS index...")
    # save_faiss_index(embeddings, chunks, document_mapping)
    # print("FAISS index saved successfully.")

    # Main interaction loop
    while True:
        print("\n" + "="*50)
        user_query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        
        if user_query.lower() == "exit":
            print("Thank you for using the Document QA System. Goodbye!")
            break

        try:
            top_k = int(input("How many document chunks should I analyze? (default: 5): ").strip() or "5")
        except ValueError:
            print("Invalid input. Using default value of 5.")
            top_k = 5

        results = search_query(user_query, top_k)
        enhanced_display_results(results, max_distance=10, query=user_query, base_path=base_path)

if __name__ == "__main__":
    main()