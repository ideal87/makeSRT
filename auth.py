import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow

def load_client_secrets() -> tuple[str, str] | None:
    """Load client_id and client_secret from client_secret.json."""
    if not os.path.exists('client_secret.json'):
        return None
    try:
        with open('client_secret.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key in ['installed', 'web']:
            if key in data:
                return data[key].get('client_id'), data[key].get('client_secret')
    except Exception:
        pass
    return None

def authenticate():
    print("\n" + "="*60)
    print("🚀 Starting Manual YouTube OAuth Flow...")
    print("="*60 + "\n")
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
    
    if not os.path.exists('client_secret.json'):
        print("❌ ERROR: 'client_secret.json' file is not found in the directory!")
        print("Please download your OAuth client credentials JSON file from Google Cloud Console,")
        print("rename it to 'client_secret.json', and place it in this folder.")
        return

    secrets = load_client_secrets()
    if not secrets or not secrets[0] or not secrets[1]:
        print("❌ ERROR: 'client_secret.json' is missing essential fields ('client_id' or 'client_secret')!")
        return

    try:
        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
        
        # Generate the authorization URL manually
        auth_url, _ = flow.authorization_url(prompt='select_account consent', access_type='offline')
        print("🔗 If your browser doesn't open automatically, please click this link:")
        print(auth_url)
        print("\nWaiting for you to log in...")

        # Reverted back to port 8090 because Google Cloud strictly validates the exact port
        creds = flow.run_local_server(
            port=8090, 
            timeout_seconds=300, 
            authorization_prompt_kwargs={'prompt': 'select_account consent', 'access_type': 'offline'}
        )
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
        print("\n✅ Authentication successful! 'token.json' has been created.")
        print("You can now close this terminal and return to the Streamlit app.")
    except Exception as e:
        print(f"\n❌ Error during authentication: {e}")

if __name__ == "__main__":
    authenticate()
