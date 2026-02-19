import sys
import logging
import os  
from pathlib import Path
from contextlib import asynccontextmanager

# 1. SETUP PATH 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# 2. CLEAN IMPORTS 
try:
    from shared_core.schemas import PredictionRequest, PredictionResponse
    # Must import via zone_3_inference because we run from root
    from zone_3_inference.app.services.prediction_service import prediction_service
except ImportError as e:
    print(f" Import Error in main.py: {e}")
    print(f"   Sys Path: {sys.path}")
    sys.exit(1)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.main")

# Load Env
load_dotenv(PROJECT_ROOT / ".env")

# 3. SECURITY: API Key Validation

API_KEY_NAME = "X-API-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Rejects requests without the correct API key."""
    expected_key = os.getenv("API_ACCESS_TOKEN")
    
    if not expected_key:
        logger.warning(" Security Alert: API_ACCESS_TOKEN not set in .env! Allowing all requests (DEV MODE).")
        return api_key

    if api_key != expected_key:
        logger.warning(f" Unauthorized access attempt.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )
    return api_key

# 4. LIFECYCLE MANAGER
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    includes Async Database Warmup!
    """
    logger.info(" Server starting up...")
    
    #1. Load sync Artifacts (Pickle files)
    # We do this first so that the advisor object exists
    success = prediction_service.load_resources()
    if not success:
        logger.critical(" FATAL: AI Models failed to load! Readiness probe will fail.")
    else:
        # 2 . Initialize Async database
        # we must wait this cause creating tables is an async I?O operation
        if prediction_service.advisor:
            await prediction_service.advisor.warmup()
            logger.info(" Async DB Warmed up and ready.")
        logger.info("Hybrid AI Brain loaded and ready")
        
    yield
    
    logger.info(" Server shutting down...")

# 5. APP DEFINITION
app = FastAPI(
    title="Adaptive Finance AI", 
    version="1.0.0",
    lifespan=lifespan
)

# --- PROBES (Reliability) ---
@app.get("/health/liveness", tags=["Health"])
async def liveness_probe():
    return {"status": "alive"}

@app.get("/health/readiness", tags=["Health"])
async def readiness_probe():
    if not prediction_service._is_ready:
        raise HTTPException(status_code=503, detail="AI Service initializing")
    return {"status": "ready"}

@app.get("/", tags=["Health"])
def root():
    return {"status": "active", "mode": "secure_async_inference"}

# --- BUSINESS LOGIC ---

@app.post(
    "/predict/v1/context",  # Standardized endpoint name
    response_model=PredictionResponse, 
    dependencies=[Depends(verify_api_key)],  #  SECURE THIS ENDPOINT
    tags=["Inference"]
)
async def get_recommendation(request: PredictionRequest):
    """
    Main Intelligence Endpoint.
    - Validates input using Pydantic (Shared Core)
    - Runs Hybrid AI (Bandit + LLM)
    - Returns actionable advice
    """
    try:
        # Pass the WHOLE request object (Pydantic model)
        return await prediction_service.predict(request)
        
    except Exception as e:
        logger.error(f" Inference Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Error")

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces (0.0.0.0) so Docker/Android can see it
    uvicorn.run(app, host="0.0.0.0", port=8000)