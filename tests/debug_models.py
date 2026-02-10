import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Setup Path to find .env
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

# 2. Check Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print(" No API Key found.")
    exit()

print(f" Using Key: {api_key[:5]}...{api_key[-5:]}")

# 3. List Models (Simpler Version)
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("\n Fetching Available Models...")
    print("-" * 40)
    
    # Just print the name, don't check capabilities
    for model in client.models.list():
        print(f" {model.name}")
        
except Exception as e:
    print(f" Error: {e}")