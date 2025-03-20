
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os


class KeyVaultClient:
    def __init__(self):
        self.key_vault_url = os.environ.get("KEY_VAULT_URL")
        if not self.key_vault_url:
            raise ValueError("KEY_VAULT_URL environment variable is required")

        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=self.key_vault_url, credential=self.credential
        )
        self.cache = {}

    def get_secret(self, secret_name, default=None):
        """Get a secret from Key Vault with caching"""
        if secret_name in self.cache:
            return self.cache[secret_name]

        try:
            secret = self.client.get_secret(secret_name)
            self.cache[secret_name] = secret.value
            return secret.value
        except Exception as e:
            print(f"Error retrieving secret {secret_name}: {str(e)}")
            return default
