import os
import json
import logging
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing document deletion request")

    try:
        # Get request data
        req_body = req.get_json()
        document_id = req_body.get("document_id")
        user_id = req_body.get("user_id")

        # Delete from search index
        success = delete_document(user_id, document_id)

        return func.HttpResponse(
            json.dumps({"success": success}), mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            mimetype="application/json",
            status_code=500,
        )
