import os
import json
import logging
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
import openai
import PyPDF2
import docx
from io import BytesIO


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing document indexing request")

    try:
        # Get request data
        req_body = req.get_json()
        document_id = req_body.get("document_id")
        user_id = req_body.get("user_id")
        filename = req_body.get("filename")
        content_type = req_body.get("content_type")
        file_data = req_body.get("file_data")  # Base64 encoded file data

        # Decode file data if needed
        # file_bytes = base64.b64decode(file_data)

        # Process the document
        chunks_with_embeddings = process_file(file_data, content_type)
        if not chunks_with_embeddings:
            return func.HttpResponse(
                json.dumps({"success": False, "error": "Failed to process document"}),
                mimetype="application/json",
                status_code=400,
            )

        # Upload chunks to search index
        success = upload_document_chunks(
            user_id, document_id, filename, chunks_with_embeddings
        )

        result = {
            "success": success,
            "document_id": document_id,
            "chunk_count": len(chunks_with_embeddings) if success else 0,
        }

        return func.HttpResponse(json.dumps(result), mimetype="application/json")

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            mimetype="application/json",
            status_code=500,
        )
