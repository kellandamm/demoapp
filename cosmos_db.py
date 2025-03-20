import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class CosmosDBClient:
    def __init__(self, app):
        self.app = app
        self.initialized = False

        # Get configuration
        self.cosmos_url = self.get_config("COSMOS_ENDPOINT")
        self.cosmos_key = self.get_config("COSMOS_KEY")
        self.cosmos_database = self.get_config("COSMOS_DATABASE", "chatapp")
        self.cosmos_container = self.get_config("COSMOS_CONTAINER", "messages")

        # Initialize client if credentials are available
        if self.cosmos_url and self.cosmos_key:
            try:
                self.client = cosmos_client.CosmosClient(
                    self.cosmos_url, credential=self.cosmos_key
                )
                self.init_database()
                self.initialized = True
                app.logger.info("Cosmos DB client initialized successfully")
            except Exception as e:
                app.logger.error(f"Failed to initialize Cosmos DB client: {str(e)}")
        else:
            app.logger.warning("Cosmos DB credentials not provided - storage disabled")

    def get_config(self, key, default=None):
        # Try to get from KeyVault first, then environment
        value = None
        if hasattr(self.app, "key_vault_client") and self.app.key_vault_client:
            value = self.app.key_vault_client.get_secret(key)

        if not value:
            import os

            value = os.environ.get(key, default)

        return value

    def init_database(self):
        """Initialize database and container"""
        try:
            # Create database if it doesn't exist
            self.database = self.client.create_database_if_not_exists(
                id=self.cosmos_database
            )

            # Create container with partition key on user_id
            self.container = self.database.create_container_if_not_exists(
                id=self.cosmos_container,
                partition_key=PartitionKey(path="/user_id"),
                offer_throughput=400,  # Minimum throughput
            )
            return True
        except exceptions.CosmosHttpResponseError as e:
            self.app.logger.error(f"Cosmos DB initialization error: {str(e)}")
            return False

    def save_chat(self, user_id, conversation_id, chat_data):
        """Save chat data to Cosmos DB"""
        if not self.initialized:
            self.app.logger.warning("Cosmos DB not initialized - chat not saved")
            return False

        try:
            # Add metadata for Cosmos
            cosmos_item = {
                "id": str(uuid.uuid4()),  # Unique ID for the document
                "user_id": str(user_id),  # Partition key
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "chat_data": chat_data,
            }

            # Save to Cosmos
            self.container.create_item(body=cosmos_item)
            return True
        except Exception as e:
            self.app.logger.error(f"Error saving chat to Cosmos: {str(e)}")
            return False

    def get_user_conversations(self, user_id, limit=50):
        """Get recent conversations for a user"""
        if not self.initialized:
            return []

        try:
            # Query for user's conversations, ordered by timestamp
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' ORDER BY c.timestamp DESC"
            items = list(
                self.container.query_items(
                    query=query,
                    enable_cross_partition_query=False,  # Using partition key
                    max_item_count=limit,
                )
            )
            return items
        except Exception as e:
            self.app.logger.error(
                f"Error retrieving conversations from Cosmos: {str(e)}"
            )
            return []

    def get_conversation_messages(self, user_id, conversation_id):
        """Get all messages for a specific conversation"""
        if not self.initialized:
            return []

        try:
            # Query for specific conversation messages
            query = f"SELECT * FROM c WHERE c.user_id = '{user_id}' AND c.conversation_id = '{conversation_id}' ORDER BY c.timestamp ASC"
            items = list(
                self.container.query_items(
                    query=query, enable_cross_partition_query=False
                )
            )

            # Extract just the chat data
            messages = []
            for item in items:
                if "chat_data" in item and "messages" in item["chat_data"]:
                    messages.extend(item["chat_data"]["messages"])

            return messages
        except Exception as e:
            self.app.logger.error(f"Error retrieving messages from Cosmos: {str(e)}")
            return []


# Usage in your app:
# cosmos_client = CosmosDBClient(app)
