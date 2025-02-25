from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from main import preprocess_documents, save_faiss_index, search_query, generate_query_embedding, generate_llm_response


app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Homepage Route
@app.route('/')
def home():
    return render_template("index.html")  # Make sure "index.html" exists in the "templates" folder

# ✅ File Upload Route
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    # Process document and generate embeddings
    text_chunks, document_mapping = preprocess_documents(file_path)
    
    if not text_chunks:
        return jsonify({"error": "No text found in document"}), 400

    print(f"✅ Processed {len(text_chunks)} chunks.")
    
    print("⚙️ Generating embeddings...")
    embeddings = generate_query_embedding(text_chunks)

    print("💾 Saving FAISS index...")
    save_faiss_index(embeddings, text_chunks, document_mapping)  # Pass all three arguments

    print("✅ FAISS index saved successfully.")
    return jsonify({"message": "File uploaded and indexed successfully"}), 200

# ✅ Query Route
@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "")

    print("📌 Received Question:", question)  # Debug print
    embeddings = None  # Initialize embeddings as None

    # Check if FAISS index is loading
    if embeddings is None:
        print("❌ FAISS Index Not Loaded!")
        return jsonify({"error": "FAISS Index not loaded"}), 500

    print("✅ FAISS Index Loaded Successfully")  # Debug print

    # Continue with the normal processing
    results = save_faiss_index(question)
    print("🔍 Search Results:", results)  # Debug print

    return jsonify({"answer": results})


# ✅ Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
