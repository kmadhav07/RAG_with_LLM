from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from transformers import AutoModel, AutoTokenizer
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
DEFAULT_PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_pdfs')
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# E5 Model configuration  ---- using the public intfloat/e5-base-v2
MODEL_NAME = "intfloat/e5-base-v2"

# Gemini API keys – multiple keys for fallback if one hits quota or fails
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY", "your-api-key"),
]

# Default sample query for the demo
DEFAULT_QUERY = "What are the key concepts in machine learning and deep learning?"

# ---------------------------------------------------------------------------
# Load E5 Model
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading E5-base-v2 Model from Hugging Face")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print("Note: First-time download ~438 MB (cached afterwards)")
print()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("[OK] Tokenizer loaded")
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("[OK] Model loaded")
    model.eval()
    print()
    print("=" * 70)
    print("Model loaded successfully!")
    print("=" * 70)
    print()
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Model download may be in progress - wait for completion")
    exit(1)

# ---------------------------------------------------------------------------
# Gemini initialisation helper (lazy, called when user activates Gemini mode)
# ---------------------------------------------------------------------------
gemini_model = None
gemini_initialised = False


def _init_gemini():
    """Initialise Gemini model if not done yet. Returns True on success.
    Tries multiple API keys and models in order until one works."""
    global gemini_model, gemini_initialised
    if gemini_initialised:
        return gemini_model is not None
    gemini_initialised = True

    import google.generativeai as genai

    # List of models to try in order
    models_to_try = [
        'models/gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
    ]

    for api_key in GEMINI_API_KEYS:
        print(f"[INFO] Trying API key ...{api_key[-6:]}")
        try:
            genai.configure(api_key=api_key)
            for model_name in models_to_try:
                try:
                    candidate = genai.GenerativeModel(model_name)
                    candidate.generate_content("test")
                    gemini_model = candidate
                    print(f"[OK] Gemini API initialised ({model_name}) with key ...{api_key[-6:]}")
                    return True
                except Exception as model_err:
                    print(f"  [WARN] {model_name} failed: {model_err}")
                    continue
        except Exception as key_err:
            print(f"  [WARN] Key ...{api_key[-6:]} configuration failed: {key_err}")
            continue

    print("[ERROR] All Gemini API keys and models failed")
    gemini_model = None
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def truncate_to_tokens(text, max_tokens=512):
    """Truncate text to the first *max_tokens* tokens (word-pieces).
    
    We tokenize, keep only the first max_tokens token-ids, then decode
    back to a string so the embedding model always sees exactly the first
    512 tokens of the document.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def get_embedding(text, prefix="query: "):
    """Get E5 embedding for text with the proper prefix."""
    text_with_prefix = prefix + text

    inputs = tokenizer(text_with_prefix, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.numpy()


def find_top_documents(query, pdf_texts, pdf_names, top_k=5):
    """Find top-k most similar documents to the query."""
    query_embedding = get_embedding(query, prefix="query: ")

    doc_embeddings = []
    for text in pdf_texts:
        truncated = truncate_to_tokens(text, max_tokens=512)
        embedding = get_embedding(truncated, prefix="passage: ")
        doc_embeddings.append(embedding)

    doc_embeddings = np.vstack(doc_embeddings)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'filename': pdf_names[idx],
            'similarity': float(similarities[idx]),
            'preview': pdf_texts[idx][:500] + '...' if len(pdf_texts[idx]) > 500 else pdf_texts[idx],
            'full_text': pdf_texts[idx]
        })

    return results


def _load_pdfs_from_folder(folder):
    """Load all PDFs from a folder. Returns (texts, names) lists."""
    texts, names = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith('.pdf'):
            fpath = os.path.join(folder, fname)
            text = extract_text_from_pdf(fpath)
            if text:
                texts.append(text)
                names.append(fname)
                print(f"  [OK] {fname} ({len(text)} chars)")
            else:
                print(f"  [SKIP] {fname} - No text found")
    return texts, names


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', default_query=DEFAULT_QUERY)


@app.route('/list-defaults', methods=['GET'])
def list_defaults():
    """Return the list of available default PDF filenames."""
    try:
        names = sorted([
            f for f in os.listdir(DEFAULT_PDF_FOLDER)
            if f.lower().endswith('.pdf')
        ])
        return jsonify({'success': True, 'files': names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load-defaults', methods=['POST'])
def load_defaults():
    """Process selected default PDF files with either the default or a custom query."""
    try:
        data = request.get_json() or {}
        query = (data.get('query') or DEFAULT_QUERY).strip()
        use_gemini = data.get('use_gemini', False)
        selected_files = data.get('selected_files', [])  # list of filenames

        if not query:
            query = DEFAULT_QUERY

        if not selected_files:
            return jsonify({'error': 'No documents selected'}), 400

        print(f"\n{'='*60}")
        print(f"Processing {len(selected_files)} selected default PDFs")
        print(f"Query: '{query}'")
        print(f"{'='*60}")

        # Load only the selected files
        pdf_texts, pdf_names = [], []
        for fname in selected_files:
            fpath = os.path.join(DEFAULT_PDF_FOLDER, fname)
            if os.path.isfile(fpath) and fname.lower().endswith('.pdf'):
                text = extract_text_from_pdf(fpath)
                if text:
                    pdf_texts.append(text)
                    pdf_names.append(fname)
                    print(f"  [OK] {fname} ({len(text)} chars)")
                else:
                    print(f"  [SKIP] {fname} - No text found")

        if not pdf_texts:
            return jsonify({'error': 'No valid PDF files found in default folder'}), 400

        print(f"\nE5 Analysis in progress...")
        top_docs = find_top_documents(query, pdf_texts, pdf_names, top_k=min(5, len(pdf_texts)))
        print(f"[OK] E5 Complete! Top: {top_docs[0]['filename']} ({top_docs[0]['similarity']:.1%})")

        # Optional Gemini summary
        gemini_summary = None
        if use_gemini and _init_gemini() and gemini_model:
            gemini_summary = _generate_gemini_summary(query, top_docs)

        # Build chat context
        chat_context = ""
        if use_gemini:
            chat_context = f"Query: {query}\n\n"
            for i, doc in enumerate(top_docs, 1):
                chat_context += f"Document {i}: {doc['filename']}\nSimilarity Score: {doc['similarity']:.1%}\n"
                chat_context += f"Content: {doc['full_text'][:3000]}\n\n"

        # Strip full_text for JSON response size
        for doc in top_docs:
            doc.pop('full_text', None)

        return jsonify({
            'success': True,
            'query': query,
            'total_documents': len(pdf_texts),
            'top_documents': top_docs,
            'source': 'default',
            'gemini_enabled': use_gemini and gemini_model is not None,
            'gemini_summary': gemini_summary,
            'chat_context': chat_context,
            'default_file_names': pdf_names,
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing."""
    try:
        query = request.form.get('query', '').strip()
        use_gemini = request.form.get('use_gemini', 'false').lower() == 'true'

        if not query:
            return jsonify({'error': 'Please provide a search query'}), 400

        files = request.files.getlist('pdfs')

        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400

        pdf_texts = []
        pdf_names = []

        # Clear upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        print(f"\n{'='*60}")
        print(f"Processing {len(files)} uploaded PDFs")
        print(f"Query: '{query}'")
        print(f"{'='*60}")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                text = extract_text_from_pdf(filepath)
                if text:
                    pdf_texts.append(text)
                    pdf_names.append(filename)
                    print(f"  [OK] {filename} ({len(text)} chars)")
                else:
                    print(f"  [SKIP] {filename} - No text found")

        if len(pdf_texts) == 0:
            return jsonify({'error': 'No valid PDF files with text found'}), 400

        print(f"\nE5 Analysis in progress...")
        top_docs = find_top_documents(query, pdf_texts, pdf_names, top_k=min(5, len(pdf_texts)))
        print(f"[OK] E5 Complete! Top: {top_docs[0]['filename']} ({top_docs[0]['similarity']:.1%})")

        # Optional Gemini summary
        gemini_summary = None
        if use_gemini and _init_gemini() and gemini_model:
            gemini_summary = _generate_gemini_summary(query, top_docs)

        # Build chat context
        chat_context = ""
        if use_gemini:
            chat_context = f"Query: {query}\n\n"
            for i, doc in enumerate(top_docs, 1):
                doc_text = pdf_texts[pdf_names.index(doc['filename'])]
                chat_context += f"Document {i}: {doc['filename']}\nSimilarity Score: {doc['similarity']:.1%}\n"
                chat_context += f"Content: {doc_text[:3000]}\n\n"

        # Strip full_text for JSON response size
        for doc in top_docs:
            doc.pop('full_text', None)

        print(f"{'='*60}\n")

        return jsonify({
            'success': True,
            'query': query,
            'total_documents': len(pdf_texts),
            'top_documents': top_docs,
            'source': 'uploaded',
            'gemini_enabled': use_gemini and gemini_model is not None,
            'gemini_summary': gemini_summary,
            'chat_context': chat_context,
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    """Handle follow-up questions about the documents."""
    try:
        if not _init_gemini() or gemini_model is None:
            return jsonify({'error': 'Gemini is not available'}), 400

        data = request.get_json()
        question = data.get('question', '').strip()
        context = data.get('context', '')

        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        print(f"\nChat Question: '{question}'")

        if context:
            prompt = f"""Based on the documents we analysed earlier:

{context}

User's follow-up question: {question}

Please provide a clear, concise answer based on the document content."""
        else:
            prompt = question

        response = gemini_model.generate_content(prompt)
        print("[OK] Response generated")

        return jsonify({
            'success': True,
            'question': question,
            'answer': response.text,
        })

    except Exception as e:
        print(f"[ERROR] Chat error: {str(e)}")
        return jsonify({'error': f'Chat error: {str(e)}'}), 500


# ---------------------------------------------------------------------------
# Gemini summary helper
# ---------------------------------------------------------------------------

def _generate_gemini_summary(query, top_docs):
    """Ask Gemini to summarise why the top documents match the query."""
    try:
        doc_info = ""
        for i, doc in enumerate(top_docs, 1):
            preview = doc.get('full_text', doc.get('preview', ''))[:2000]
            doc_info += f"\n--- Document {i}: {doc['filename']} (Similarity: {doc['similarity']:.1%}) ---\n{preview}\n"

        prompt = f"""The user searched for: "{query}"

The following documents were found to be most relevant (ranked by cosine similarity of E5-base-v2 embeddings):
{doc_info}

Please provide:
1. A brief summary of why these documents are relevant to the query.
2. Key topics covered across the top results.
Keep the response concise (max 200 words)."""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[WARN] Gemini summary failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Serve PDFs
# ---------------------------------------------------------------------------

@app.route('/serve-pdf/<source>/<filename>')
def serve_pdf(source, filename):
    """Serve a PDF file for viewing."""
    if source == 'default':
        folder = DEFAULT_PDF_FOLDER
    elif source == 'uploaded':
        folder = os.path.abspath(UPLOAD_FOLDER)
    else:
        return 'Invalid source', 404
    safe_name = secure_filename(filename)
    return send_from_directory(folder, safe_name, mimetype='application/pdf')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting Flask Web Server...")
    print("=" * 70)
    print("Server URL: http://localhost:5000")
    print("Local IP:   http://0.0.0.0:5000")
    print()
    print(f"Default PDFs folder: {DEFAULT_PDF_FOLDER}")
    num_defaults = len([f for f in os.listdir(DEFAULT_PDF_FOLDER) if f.lower().endswith('.pdf')]) if os.path.isdir(DEFAULT_PDF_FOLDER) else 0
    print(f"Default PDFs available: {num_defaults}")
    print()
    print("Open your browser and navigate to the URL above")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
