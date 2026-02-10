
from setuptools import setup, find_packages

setup(
    name="adaptive_finance_ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # --- CORE MATH & ML ---
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        
        # --- SERVER (Zone 3) ---
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",  # [standard] includes uvloop for speed
        "pydantic>=2.0.0",
        "pydantic-settings",          #Needed for config.py
        
        # --- AI & UTILS ---
        "python-dotenv",
        "google-genai",    # The new V1 SDK
        "tenacity",        # For smart retries
        
        
        # --- VISUALIZATION (Zone 1) ---
        "matplotlib",
        "seaborn"
    ],
)