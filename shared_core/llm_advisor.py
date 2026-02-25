
import os
import json
import time
import random
import logging
import asyncio
import aiosqlite
from pathlib import Path
from typing import Dict, Optional

# Tenacity for smart retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup Logger
logger = logging.getLogger(__name__)

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_NEW_SDK = True
except ImportError:
    HAS_NEW_SDK = False
    logger.warning(" 'google-genai' not found. Run: pip install google-genai")

# ============================================================================
# aiosqlite CACHE (Persistent, Crash-Safe)
# ============================================================================
class AsyncSQLiteCache:
    """
    Async Cache.
    Non-blocking I/O allows high concurrency (10k+ req/sec).
    """
    def __init__(self, db_path: str = "llm_cache.db", max_size: int = 1000):
        self.db_path = db_path
        self.max_size = max_size
        self._init_lock = asyncio.Lock() # prevents race conditions during init
        

    async def init_db(self):
        """Async Initialization using WAL mode for performance."""
        async with self._init_lock:

            try:
                # ensure directory exists
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                
                async with aiosqlite.connect(self.db_path) as db:
                    # WAL Mode = write Ahead Logging 
                    await db.execute("PRAGMA journal_mode=WAL;")

                    #1 The Ephemeral Cache table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS cache (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            last_access REAL
                        )
                    """)
                    # Create an index for fast LRU sorting
                    await db.execute("CREATE INDEX IF NOT EXISTS idx_access ON cache(last_access)")

                    # 2 permanent analytics table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS analytics_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            prediction_id TEXT,
                            action_name TEXT,
                            reward REAL,
                            timestamp REAL,
                            uncertainty REAL DEFAULT 0.0
                        )
                    """)
                    await db.commit()
                logger.info(f" Async Cache Ready (WAL Mode): {self.db_path}")
            except Exception as e:
                logger.error(f" Cache Init Failed: {e}")

    async def get(self, key: str) -> Optional[str]:
        """Fetch value without blocking the Event Loop."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT value FROM cache WHERE key = ?", (key,)) as cursor:
                    row =  await cursor.fetchone()
                
                if row:
                    # Update LRU timestamp asynchronously fire and forget sty;e
                    await db.execute("UPDATE cache SET last_access = ? WHERE key = ?", (time.time(), key))
                    await db.commit()
                    return row[0]
                return None
        except Exception:
            return None

    async def put(self, key: str, value: str):
        """Insert value and manage eviction asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                now = time.time()
                # 1. Insert or Replace (Upsert)
                await db.execute(
                    "INSERT OR REPLACE INTO cache (key, value, last_access) VALUES (?, ?, ?)",
                    (key, value, now)
                )
                
                # 2. Check Size (Fast count)
                async with db.execute("SELECT COUNT(*) FROM cache") as cursor:
                    row = await cursor.fetchone()
                    count = row[0] if row else 0

                # 3. Evict Oldest if full
                if count > self.max_size:
                    # Delete oldest
                    await db.execute("""
                        DELETE FROM cache WHERE key = (
                            SELECT key FROM cache ORDER BY last_access ASC LIMIT 1
                        )
                    """)
                await db.commit()
        except Exception as e:
            logger.warning(f" Cache Write Failed: {e}")

    

# ============================================================================
# LLM STRATEGY ADVISOR (With Circuit Breaker & Async DB)
# ============================================================================
class LLMStrategyAdvisor:
    VOLATILITY_THRESHOLD = 0.8
    RETURN_THRESHOLD = 0.15
    MODEL_ID = 'gemini-2.5-flash-lite'
    
    # Circuit Breaker Config
    CB_FAILURE_THRESHOLD = 3      # Failures before opening circuit
    CB_RESET_TIMEOUT = 30         # Seconds to wait before trying again
    CB_LATENCY_LIMIT = 2.0        # Max allowed latency (seconds)

    def __init__(self, api_key: Optional[str] = None, cache_size: int = 1000, skip_api_init: bool = False):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Determine database path relative to this file
        base_path = Path(__file__).resolve().parent.parent
        db_path = base_path / "zone_2_artifacts" / "llm_cache.db"
        
        # Initialize SQLite Cache
        self.cache = AsyncSQLiteCache(db_path=str(db_path), max_size=cache_size)
        
        self.enabled = False
        self.client = None
        
        # Circuit Breaker State
        self._cb_failures = 0
        self._cb_open_time = 0
        self._cb_is_open = False

        if not self.api_key:
            logger.error(" Missing GEMINI_API_KEY")
            return 

        if not skip_api_init and HAS_NEW_SDK:
            self._init_api()

    def _init_api(self):
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.enabled = True
            logger.info(f" Gemini Client Initialized ({self.MODEL_ID})")
        except Exception as e:
            logger.error(f" Init Failed: {e}")
            self.enabled = False
    
    async def WARMUP(self):
        """
            Must be called during startup
            initializes the async database tables.
        """
        await self.cache.init_db()

    def _check_circuit_breaker(self) -> bool:
        """Returns True if request allowed, False if Circuit Open (Blocked)."""
        if self._cb_is_open:
            elapsed = time.time() - self._cb_open_time
            if elapsed > self.CB_RESET_TIMEOUT:
                logger.info(" Circuit Breaker: HALF-OPEN (Retrying API...)")
                self._cb_is_open = False
                self._cb_failures = 0
                return True # Allow retry
            return False # Still blocked
        return True

    def _record_failure(self, reason: str):
        """Increments failure count and opens circuit if threshold reached."""
        self._cb_failures += 1
        logger.warning(f" API Issue ({self._cb_failures}/{self.CB_FAILURE_THRESHOLD}): {reason}")
        
        if self._cb_failures >= self.CB_FAILURE_THRESHOLD:
            self._cb_is_open = True
            self._cb_open_time = time.time()
            logger.critical(f"🔌 Circuit Breaker OPENED. Pausing LLM calls for {self.CB_RESET_TIMEOUT}s.")

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _call_gemini_async(self, prompt: str, max_tokens: int = 60, temp: float = 0.7) -> Optional[str]:
        """
        Async LLM call with Latency Check & Circuit Breaker.
        """
        if not self.enabled: return None
        
        # 1. Check Circuit
        if not self._check_circuit_breaker():
            logger.info("⏭ Circuit Open: Skipping LLM call (using fallback).")
            return None

        start_time = time.time()
        
        try:
            # 2. Call API (Async)
            response = await self.client.aio.models.generate_content(
                model=self.MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp, 
                    max_output_tokens=max_tokens
                )
            )
            
            # 3. Check Latency
            duration = time.time() - start_time
            if duration > self.CB_LATENCY_LIMIT:
                self._record_failure(f"Latency {duration:.2f}s > {self.CB_LATENCY_LIMIT}s")
                # Even though it succeeded, it was too slow. We use the result but warn the system.
            else:
                # Success & Fast: Reset failure count
                if self._cb_failures > 0:
                    self._cb_failures = 0
            
            return response.text.strip() if response.text else None

        except Exception as e:
            self._record_failure(f"Exception: {str(e)}")
            # retry (unless circuit opens next time)
            raise e

    # ------------------------------------------------------------------
    # PUBLIC METHODS fully async aware
    # ------------------------------------------------------------------

    async def get_strategy_recommendation(self, user_features: Dict) -> str:
        vol = user_features.get('spending_volatility', 0)
        ret = user_features.get('return_rate', 0)
        
        # Cache Key
        cache_key = json.dumps({'t': 'strat', 'v': round(vol, 1), 'r': round(ret, 1)}, sort_keys=True)
        #Await the get
        if cached := await self.cache.get(cache_key): 
            return cached

        # Fallback if Circuit Open or Disabled
        if not self.enabled or (self._cb_is_open and not self._check_circuit_breaker()):
            return "EXPLORE" if vol > self.VOLATILITY_THRESHOLD else "EXPLOIT"

        prompt = f"""
        User Volatility: {vol:.2f} (Threshold {self.VOLATILITY_THRESHOLD})
        User Returns: {ret:.2f} (Threshold {self.RETURN_THRESHOLD})
        Rule: EXPLORE if Volatility > {self.VOLATILITY_THRESHOLD} OR Returns > {self.RETURN_THRESHOLD}. Else EXPLOIT.
        Output exactly one word: EXPLORE or EXPLOIT.
        """
        
        try:
            res = await self._call_gemini_async(prompt, max_tokens=10, temp=0.0)
            final = "EXPLORE" if (res and "EXPLORE" in res.upper()) else "EXPLOIT"

            # Await the PUT
            await self.cache.put(cache_key, final)
            return final
        except Exception:
            # Final safety net if all retries fail
            return "EXPLOIT"

    async def generate_message(self, user_features: Dict, action_name: str) -> str:
        vol = user_features.get('spending_volatility', 0)
        
        # Vibe Check
        vibes = ["witty", "serious", "friendly", "urgent", "calm"]
        current_vibe = random.choice(vibes)

        # Cache Key (Includes Vibe for variety)
        cache_key = json.dumps({'t': f'msg_{action_name}', 'v': round(vol, 1), 'mood': current_vibe}, sort_keys=True)
        # AWAIT THE GET
        if cached :=  await self.cache.get(cache_key):
            return cached

        # Fallback Logic (Instant response if LLM is down)
        fallbacks = {
            "strict_budget": "⚠️ Spending alert! Time to tighten the budget.",
            "streak_builder": "🔥 Keep your streak alive! Save today.",
            "quiz": "🧠 Quick finance quiz available!",
            "cool_off": "🧊 Freeze spending for 24 hours."
        }
        
        if not self.enabled or (self._cb_is_open and not self._check_circuit_breaker()):
            return fallbacks.get(action_name, "Check your financial health!")

        # Dynamic Prompt
        risk_context = "HIGH" if vol > 0.8 else "STABLE"
        prompt = f"""
        You are an AI Assistant for a Student Budget App.
        The user has {risk_context} spending volatility ({vol:.2f}).
        
        TASK:
        Write a 1-sentence notification (MAX 12 WORDS) triggering the action: "{action_name}".
        
        CONTEXT:
        - This is about DAILY CASH, NOT stock portfolios.
        - Tone: {current_vibe.upper()}.
        - Use exactly one emoji.
        
        Output strictly the message text:
        """

        try:
            msg = await self._call_gemini_async(prompt, max_tokens=60, temp=0.9)
            if msg:
                # AWAIT THE PUT
                await self.cache.put(cache_key, msg)
                return msg
            return fallbacks.get(action_name, "Check your financial health! 📉")
        except Exception:
            return fallbacks.get(action_name, "Check your financial health! 📉")