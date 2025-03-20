import os
import json
import logging
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import openai


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing document search request")

    try:
        # Get request data
        req_body = req.get_json()
        query = req_body.get("query")
        user_id = req_body.get("user_id")
        document_ids = req_body.get("document_ids", [])

        # Get embedding for the query
        query_embedding = get_embedding(query)

        # Search for relevant content
        search_results = search_documents(user_id, query_embedding, top_k=5)

        return func.HttpResponse(
            json.dumps({"success": True, "results": search_results}),
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Error searching documents: {str(e)}")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            mimetype="application/json",
            status_code=500,
        )
