# helpers.py

def process_file(file_data, content_type):
    """Process a file and return chunks with embeddings"""
    # Extract text from file
    text = extract_text(file_data, content_type)
    if not text:
        return []
    
    # Split text into chunks
    text_chunks = chunk_text(text)
    
    # Get embeddings for chunks
    return get_embeddings(text_chunks)

def extract_text(file_data, content_type):
    """Extract text from various file types"""
    # Implementation similar to your DocumentProcessor.extract_text method
    
def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into chunks with overlap"""
    # Implementation similar to your DocumentProcessor.chunk_text method
    
def get_embeddings(text_chunks):
    """Get embeddings for text chunks"""
    # Implementation similar to your DocumentProcessor.get_embeddings method
    
def get_embedding(text):
    """Get embedding for a single text"""
    # Implementation

def upload_document_chunks(user_id, document_id, filename, chunks):
    """Upload document chunks to the search index"""
    # Implementation similar to your AISearchManager.upload_document_chunks method
    
def search_documents(user_id, query_embedding, top_k=5):
    """Search documents by vector similarity"""
    # Implementation similar to your AISearchManager.search_documents method
    
def delete_document(user_id, document_id):
    """Delete all chunks for a document"""
    # Implementation similar to your AISearchManager.delete_document method