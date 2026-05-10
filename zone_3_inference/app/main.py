import sys
import logging
import os  
from pathlib import Path
from collections import defaultdict
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import HTTPException

# 1. SETUP PATH 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Depends, Security, status,BackgroundTasks,Request
from fastapi.security import APIKeyHeader
from fastapi.templating import Jinja2Templates
import aiosqlite
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# Load Env
load_dotenv(PROJECT_ROOT / ".env")

# 2. CLEAN IMPORTS 
try:
    from shared_core.schemas import PredictionRequest, PredictionResponse, FeedbackRequest,LeaderboardUpdateRequest,CheerRequest
    # Must import via zone_3_inference because we run from root
    from zone_3_inference.app.services.prediction_service import prediction_service
    from zone_3_inference.app.services.leaderboard_service import LeaderboardService

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

    #Grab the smartest brain from the cloud before we load anything!
    prediction_service.download_latest_brain()

    #1. Load sync Artifacts (Pickle files)
    # We do this first so that the advisor object exists
    success = prediction_service.load_resources()
    if not success:
        logger.critical(" FATAL: AI Models failed to load! Readiness probe will fail.")
    else:
        # 2 . Initialize Async database
        # we must wait this cause creating tables is an async I?O operation
        if prediction_service.advisor:
            await prediction_service.advisor.WARMUP()
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
# Setup Templates Directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# parse origin from .env 
origin_str = os.getenv("ALLOWED_ORIGIN", "")
allowed_origins = [origin.strip() for origin in origin_str.split(",") if origin.strip()]

if not allowed_origins:
    logger.warning("No ALLOWED_ORIGINS set. Defaulting to strict local only.")
    allowed_origins = ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Token", "Content-Type"], 
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
    

@app.post (
    "/predict/v1/feedback",
    dependencies=[Depends(verify_api_key)],
    tags=["Inference"],
    status_code=status.HTTP_202_ACCEPTED
)

async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks ):
    """
    Asynchronous Learning Loop.
    Instantly returns to the Android app, while updating the bandit in the background
    """
    #add math and disk operations to back the background queue
    background_tasks.add_task(prediction_service.process_feedback, request)

    return {
        "status": "accepted", 
        "message": f"Feedback for {request.prediction_id} queued for processing"
    }


# --- GAMIFICATION and LEADERBOARD ENDPOINTS ---

@app.post("/gamification/v1/leaderboard/update", dependencies=[Depends(verify_api_key)])
async def sync_xp(request: LeaderboardUpdateRequest):
    """Android app calls this to silently back up the user's XP and Tier."""
    try:
        service = LeaderboardService()
        data = service.update_user_xp(
            user_id=request.user_id,
            anonymous_name=request.anonymous_name,
            xp=request.xp,
            tier=request.tier
        )
        return {"status": "success", "message": "Leaderboard updated"}
    except Exception as e:
        logger.error(f"Sync XP Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/gamification/v1/leaderboard/top", dependencies=[Depends(verify_api_key)])
async def get_leaderboard():
    """Android app calls this to display the community Leaderboard."""
    try:
        service = LeaderboardService()
        top_users = service.get_top_50()
        return {"status": "success", "data": top_users}
    except Exception as e:
        logger.error(f"Fetch Leaderboard Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gamification/v1/leaderboard/cheer", dependencies=[Depends(verify_api_key)])
async def send_cheer(request: CheerRequest):
    """Endpoint triggered by Android Double-Tap gesture."""
    try:
        service = LeaderboardService()
        return service.register_cheer(request.target_user_id)
    except Exception as e:
        logger.error(f"Cheer Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to register cheer")
    

    
@app.get("/gamification/v1/leaderboard/history", dependencies=[Depends(verify_api_key)], tags=["Gamification"])
async def get_leaderboard_history():
    """Returns Hall of Fame winners for the Android UI."""
    try:
        service = LeaderboardService()
        return {"status": "success", "data": service.get_historical_winners()}
    except Exception as e:
        logger.error(f"Hall of Fame Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/admin/dashboard", tags=["Monitoring"])
async def get_dashboard(request: Request):
    """Generate a live HTML dashboard pulling from Supabase Cloud."""
    labels, counts, success_rates = ["No Data"], [1], [0]
    trend_labels, trend_ctr, trend_uncertainty, cumulative_rewards = [], [], [], []
    context_data = [] 
    error_msg = None

    try:
        #1. Fetch data from Supabase instead of SQLite
        if prediction_service.supabase:
            response = prediction_service.supabase.table("analytics_log").select("action_name, reward,created_at, uncertainty","user_segment").execute()
            data = response.data
            
            if data:
                #A Calculate Action Distribution and CTR 
                action_counts = defaultdict(int)
                action_rewards = defaultdict(list)
                
                #B Calculate Timeline Grouping 
                daily_stats = defaultdict(lambda: {"rewards": [], "uncertainties": [], "clicks": 0})
                
                for row in data:
                    # Tally Actions
                    action = row.get("action_name", "unknown")
                    action_counts[action] += 1
                    action_rewards[action].append(row.get("reward", 0))
                    
                    # Parse Time (Handles both Float timestamps and Supabase ISO strings)
                    raw_time = row.get("created_at")
                    if raw_time:
                        if isinstance(raw_time, (int, float)):
                            day = datetime.fromtimestamp(raw_time).strftime('%Y-%m-%d')
                        else:
                            day = str(raw_time)[:10] # Slices '2026-02-25T13:43:46' to '2026-02-25'
                        
                        daily_stats[day]["rewards"].append(row.get("reward", 0))
                        daily_stats[day]["uncertainties"].append(row.get("uncertainty", 0.0) or 0.0)
                        daily_stats[day]["clicks"] += row.get("reward", 0)

                
                #B.5 Calculate Contextual Segment Performance
                context_stats = defaultdict(lambda: {"rewards": []})
                
                for row in data:
                    action = row.get("action_name", "unknown")
                    segment = row.get("user_segment", "Stable") #Default to Stable if missing
                    if segment:
                        context_stats[(action, segment)]["rewards"].append(row.get("reward", 0))

                context_data = []
                for (action, segment), stats in context_stats.items():
                    if stats["rewards"]:
                        ctr = round((sum(stats["rewards"]) / len(stats["rewards"])) * 100, 1)
                        context_data.append({"action": action, "segment": segment, "ctr": ctr})


                # Format Action Data
                labels = list(action_counts.keys())
                counts = [action_counts[label] for label in labels]
                success_rates = [
                    round((sum(action_rewards[label]) / len(action_rewards[label])) * 100, 1) 
                    for label in labels
                ]

                #C Format Timeline Data 
                sorted_days = sorted(daily_stats.keys()) # Sort chronologically
                trend_labels = sorted_days
                
                for day in sorted_days:
                    # Daily CTR
                    day_rewards = daily_stats[day]["rewards"]
                    if day_rewards:
                        trend_ctr.append(round((sum(day_rewards) / len(day_rewards)) * 100, 1))
                    else:
                        trend_ctr.append(0)
                        
                    # Daily Average Uncertainty
                    day_uncerts = daily_stats[day]["uncertainties"]
                    if day_uncerts:
                        trend_uncertainty.append(round(sum(day_uncerts) / len(day_uncerts), 4))
                    else:
                        trend_uncertainty.append(0)
                        
                #D Calculate Cumulative Rewards 
                current_total = 0
                for day in sorted_days:
                    current_total += daily_stats[day]["clicks"]
                    cumulative_rewards.append(current_total)
        else:
            error_msg = "Supabase connection not initialized. Check .env keys."

    except Exception as e:
        logger.error(f"Dashboard Database Error: {e}")
        error_msg = f"Could not load analytics from cloud: {str(e)}"

    # Pass data securely to the HTML file
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "labels": labels,
            "counts": counts,
            "success_rates": success_rates,
            "trend_labels": trend_labels,                 
            "trend_ctr": trend_ctr,                       
            "trend_uncertainty": trend_uncertainty,       
            "cumulative_rewards": cumulative_rewards,
            "context_data": context_data,
            "error_msg": error_msg
        }
    )


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces (0.0.0.0) so Docker/Android can see it
    uvicorn.run(app, host="0.0.0.0", port=8000)