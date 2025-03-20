# auth_config.py
import os
from dotenv import load_dotenv

load_dotenv()

# App Registration details
CLIENT_ID = os.getenv("ENTRA-CLIENT-ID")
CLIENT_SECRET = os.getenv("ENTRA-CLIENT-SECRET")
TENANT_ID = os.getenv("ENTRA-TENANT-ID")


# Authority and endpoints
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
REDIRECT_PATH = "/auth/callback"
ENDPOINT = "https://graph.microsoft.com/v1.0/me"

# User profile attributes to request
SCOPE = ["User.Read", "User.ReadBasic.All", "Directory.ReadWrite.All"]

# Session config
SESSION_TYPE = "filesystem"

# App config
ALLOWED_DOMAINS = ["microsoft.com"]  # Optional domain restriction
ADMIN_EMAILS = ["kedamm@microsoft.com"]  # Admin email addresses

# Registration policy
AUTO_APPROVE_DOMAIN = False  # Auto-approve users from allowed domains
REQUIRE_ADMIN_APPROVAL = True  # Require admin approval for all new users
