import hmac
import hashlib
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Chunk size for reading large files (1MB)
CHUNK_SIZE = 1024 * 1024

def load_signing_key(root_path: Path) -> bytes:
    """
    Loads the signing key from environment variables securely.
    """
    env_path = root_path / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded configuration from {env_path.name}")
    else:
        logger.warning("No .env file found; checking system environment variables.")

    key = os.getenv("MODEL_SIGNING_KEY")
   
    if not key:
        logger.error("FATAL: MODEL_SIGNING_KEY not found in env!")
        sys.exit(1)
    #stripe whitespaces from CI?CD secrets
    return key.strip().encode()

def sign_file(file_path: Path, key: bytes) -> bool:
    """
    Generates a HMAC-SHA256 signature file (.sig) for the given input file.
    Uses chunked reading to handle large models efficiently.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    try:
        logger.info(f"Signing {file_path.name}...")
        
        # Initialize HMAC with SHA256
        h = hmac.new(key, digestmod=hashlib.sha256)
        
        # Stream file in chunks to avoid RAM spikes
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                h.update(chunk)
        
        signature = h.hexdigest()
        
        # Save signature: model.pkl -> model.pkl.sig
        # Note: with_suffix replaces the extension, so we append manually
        sig_path = file_path.with_name(file_path.name + ".sig")
        
        with open(sig_path, "w") as f:
            f.write(signature)
            
        logger.info(f"Success! Signature saved to: {sig_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to sign {file_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Adaptive Finance AI - Model Signer")
    parser.add_argument("files", nargs="*", type=Path, help="Specific files to sign (optional)")
    parser.add_argument("--artifacts-dir", type=Path, default="zone_2_artifacts", help="Directory to scan for default artifacts")
    args = parser.parse_args()

    # Resolve Project Root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Load Key
    secret_key = load_signing_key(project_root)
    
    # Determine files to sign
    files_to_sign = []
    
    if args.files:
        # User provided specific files via CLI
        files_to_sign = args.files
    else:
        # Default mode: Sign known artifacts
        artifacts_path = project_root / args.artifacts_dir
        logger.info(f"No files provided. Scanning default artifacts dir: {artifacts_path}")
        
        defaults = ["bandit_model.pkl", "cmab_preprocessor.pkl"]
        # only add files that actually exist
        files_to_sign = [artifacts_path / f for f in defaults if (artifacts_path / f).exists()]

        if not files_to_sign:
            logger.warning("No default artifacts found to sign.")

    # Execution Loop
    print("-" * 40)
    success_count = 0
    for p in files_to_sign:
        if sign_file(p, secret_key):
            success_count += 1
    print("-" * 40)
    
    if success_count == len(files_to_sign) and success_count > 0:
        logger.info("All files signed successfully.")
    else:
        logger.warning(f"Signed {success_count}/{len(files_to_sign)} files. Check logs for errors.")

if __name__ == "__main__":
    main()