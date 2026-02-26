
import os
import json
import time
import random
import logging
import asyncio
import aiosqlite
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import itertools

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
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        
        self.or_models = [
            "mistralai/mistral-small-3.2-24b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-3-4b-it:free",
            "google/gemma-3-12b-it:free"
        ]
        
        base_path = Path(__file__).resolve().parent.parent
        db_path = base_path / "zone_2_artifacts" / "llm_cache.db"
        self.cache = AsyncSQLiteCache(db_path=str(db_path), max_size=cache_size)
        
        self.enabled = False
        self.client = None
        
        self._cb_failures = 0
        self._cb_open_time = 0
        self._cb_is_open = False

        if not self.api_key:
            logger.error(" Missing GEMINI_API_KEY")
        elif not skip_api_init and HAS_NEW_SDK:
            self._init_api()

        # Initialize the True Round-Robin Router once at startup
        self._build_router_pool()

    def _build_router_pool(self):
        """Builds a deterministic, 50/50 weighted infinite iterator."""
        pool = []
        or_entries = [f"openrouter:{m}" for m in self.or_models] if self.or_key else []

        if self.api_key:
            if or_entries:
                # Weight Gemini to perfectly match the number of OpenRouter models
                pool.extend(["gemini_sdk"] * len(or_entries))
            else:
                pool.append("gemini_sdk")

        pool.extend(or_entries)

        if not pool:
            self.router = None
            self.pool_size = 0
            return

        # Shuffle once to interleave Gemini and OpenRouter 
        random.shuffle(pool) 
        self.router = itertools.cycle(pool) #Create the infinite loop
        self.pool_size = len(pool)
        logger.info(f" True Round-Robin Router initialized with {self.pool_size} nodes.")

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
        
    #RAW HTTP OPENROUTER LOGIC
    def _call_openrouter_raw(self, prompt: str, model_id: str, max_tokens: int, temp: float) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.or_key}", "Content-Type": "application/json"}
        data = {
            "model": model_id, 
            "messages": [{"role": "user", "content": prompt}], 
            "max_tokens": max_tokens, "temperature": temp
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=4) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip().strip('"')


#True Weighted Round-Robin Orchestrator
    async def _multiplex_generate(self, prompt: str, max_tokens: int = 60, temp: float = 0.7) -> Optional[str]:
        if not hasattr(self, 'router') or not self.router:
            logger.error("No LLM providers available in the routing pool!")
            return None

        attempts = 0
        max_retries = self.pool_size

        # Waterfall through the pre-calculated sequence
        while attempts < max_retries:
            route = next(self.router) # O(1) instantaneous rotation
            attempts += 1

            try:
                if route == "gemini_sdk":
                    # Dynamic fast-fail: Skip Gemini instantly if the circuit breaker is open
                    if self._cb_is_open and not self._check_circuit_breaker():
                        logger.debug("Skipping Gemini (Circuit Open), rotating...")
                        continue

                    res = await self._call_gemini_async(prompt, max_tokens, temp)
                    if res:
                        logger.info("Round-Robin: Routed via Gemini SDK")
                        return res

                elif route.startswith("openrouter:"):
                    model_id = route.split("openrouter:")[1]
                    res = await asyncio.to_thread(self._call_openrouter_raw, prompt, model_id, max_tokens, temp)
                    if res:
                        logger.info(f"Round-Robin: Routed via OpenRouter ({model_id})")
                        return res

            except Exception as e:
                logger.debug(f"Route {route} failed: {e}. Rotating to next provider...")
                continue

        logger.error("Orchestrator failed: All APIs in the rotation are down or rate-limited.")
        return None

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
        
        res = await self._multiplex_generate(prompt, max_tokens=10, temp=0.0)
        final = "EXPLORE" if (res and "EXPLORE" in res.upper()) else "EXPLOIT"

        await self.cache.put(cache_key, final)
        return final

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
        
        
        # Dynamic Prompt
        risk_context = "HIGH" if vol > 0.8 else "STABLE"
        prompt = f"""
        You are an AI Assistant for a Gamified Personal Finance App.
        The user has {risk_context} spending volatility ({vol:.2f}).
        
        TASK:
        Write a 1-sentence notification (MAX 12 WORDS) triggering the action: "{action_name}".
        
        CONTEXT:
        - This is about DAILY CASH, NOT stock portfolios.
        - Focus purely on their spending behavior, not their age or profession.
        - Tone: {current_vibe.upper()}.
        - Use exactly one emoji.
        
        Output strictly the message text:
        """

        msg = await self._multiplex_generate(prompt, max_tokens=60, temp=0.9)
        final_msg = msg if msg else fallbacks.get(action_name, "Check your financial health! 📉")
        
        await self.cache.put(cache_key, final_msg)
        return final_msg