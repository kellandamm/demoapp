import os
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv

load_dotenv()

# Try to import Azure libraries
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings

    azure_available = True
except ImportError:
    azure_available = False


class FileStorage:
    """Wrapper for file storage that can use Azure Blob or local filesystem"""

    def __init__(self, app, key_vault_client=None):
        self.app = app
        self.use_azure = False

        # Try to get credentials from KeyVault first, then fall back to environment variables
        connection_string = None
        container_name = "userfiles"

        if key_vault_client:
            connection_string = key_vault_client.get_secret(
                "AZURE-STORAGE-CONNECTION-STRING"
            )
            container_name_from_kv = key_vault_client.get_secret(
                "AZURE-STORAGE-CONTAINER-NAME"
            )
            if container_name_from_kv:
                container_name = container_name_from_kv

        # Fall back to environment variables if not in KeyVault
        if not connection_string:
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            container_name = os.environ.get(
                "AZURE_STORAGE_CONTAINER_NAME", container_name
            )

        # Try to initialize Azure if available and configured
        if azure_available and connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    connection_string
                )
                self.container_name = container_name
                self._ensure_container_exists()
                self.use_azure = True
                app.logger.info("Using Azure Blob Storage for files")
            except Exception as e:
                app.logger.error(f"Failed to initialize Azure Blob Storage: {str(e)}")
                app.logger.info("Falling back to local file storage")
        else:
            if not azure_available:
                app.logger.warning(
                    "Azure Storage SDK not installed. Using local storage."
                )
            elif not connection_string:
                app.logger.warning(
                    "Azure Storage connection string not set. Using local storage."
                )

    def _ensure_container_exists(self):
        """Ensure the Azure container exists"""
        if not self.use_azure:
            return

        try:
            self.blob_service_client.get_container_client(
                self.container_name
            ).get_container_properties()
        except Exception:
            self.blob_service_client.create_container(self.container_name)

    def _ensure_user_dirs(self, user_id):
        """Ensure user directories exist for local storage"""
        base_dir = os.path.join(self.app.config["BASE_DIR"], str(user_id))
        uploads_dir = os.path.join(base_dir, "uploads")

        os.makedirs(uploads_dir, exist_ok=True)

        return uploads_dir

    def upload_file(self, file_stream, filename, user_id):
        """Upload a file to storage"""
        secure_name = secure_filename(filename)

        if self.use_azure:
            # Azure Blob Storage upload
            blob_name = f"{user_id}/{secure_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            content_settings = ContentSettings(content_type=file_stream.content_type)
            blob_client.upload_blob(
                file_stream, overwrite=True, content_settings=content_settings
            )
            return blob_name
        else:
            # Local file upload
            uploads_dir = self._ensure_user_dirs(user_id)
            filepath = os.path.join(uploads_dir, secure_name)
            file_stream.save(filepath)
            return filepath

    def download_file(self, filename, user_id):
        """Download a file from storage"""
        secure_name = secure_filename(filename)

        if self.use_azure:
            # Azure Blob Storage download
            blob_name = f"{user_id}/{secure_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            blob_data = blob_client.download_blob()
            return blob_data.readall()
        else:
            # Local file download
            uploads_dir = self._ensure_user_dirs(user_id)
            filepath = os.path.join(uploads_dir, secure_name)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    return f.read()
            return None

    def delete_file(self, filename, user_id):
        """Delete a file from storage"""
        secure_name = secure_filename(filename)

        if self.use_azure:
            # Azure Blob Storage delete
            blob_name = f"{user_id}/{secure_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            blob_client.delete_blob()
        else:
            # Local file delete
            uploads_dir = self._ensure_user_dirs(user_id)
            filepath = os.path.join(uploads_dir, secure_name)
            if os.path.exists(filepath):
                os.remove(filepath)

    def list_files(self, user_id):
        """List all files for a user"""
        if self.use_azure:
            # Azure Blob Storage list
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            blobs = container_client.list_blobs(name_starts_with=f"{user_id}/")

            files = []
            for blob in blobs:
                # Get just the filename without the user_id prefix
                if "/" in blob.name:
                    filename = blob.name.split("/", 1)[1]
                    files.append(
                        {
                            "name": filename,
                            "size": blob.size,
                            "modified": blob.last_modified.timestamp()
                            if hasattr(blob.last_modified, "timestamp")
                            else 0,
                            "blob_name": blob.name,
                        }
                    )

            return files
        else:
            # Local file list
            uploads_dir = self._ensure_user_dirs(user_id)
            files = []

            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    files.append(
                        {
                            "name": filename,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path),
                            "filepath": file_path,
                        }
                    )

            return files
