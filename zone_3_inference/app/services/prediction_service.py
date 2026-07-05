
import pickle
import asyncio
import hmac
import hashlib
import os
import logging
import json
import uuid
import numpy as np
from supabase import create_client,Client 
from pathlib import Path
from typing import Optional, Dict

#for dashboard logging


# CLEAN IMPORTS (Assumes 'pip install -e .' was run)
try:
    from shared_core.llm_advisor import LLMStrategyAdvisor
    from shared_core.models import LinUCB
    from shared_core.preprocessing import BanditPreprocessor
    from shared_core.schemas import PredictionRequest, PredictionResponse, FeedbackRequest
except ImportError as e:
    print(f" Package Import Error: {e}. Ensure you ran 'pip install -e .'")
    raise e

# Setup Logger
logger = logging.getLogger(__name__)

# Custom Exception for Security
class ModelIntegrityError(Exception):
    """Raised when a model file fails the HMAC signature check."""
    pass

class PredictionService:
    def __init__(self):
        self.bandit: Optional[LinUCB] = None
        self.preprocessor: Optional[BanditPreprocessor] = None
        self.advisor: Optional[LLMStrategyAdvisor] = None
        
        # Action Map (Must match Trainer) loads from training_config.json
        self.actions : Dict[int,str] = {}
        self._is_ready = False
        
        # SECURITY: Load key from env 
        key = os.getenv("MODEL_SIGNING_KEY")
        if not key:
            raise RuntimeError ("CRITICAL: MODEL_SIGNING_KEY not set! Server refuses to start.")
        self.secret_key = key.encode()

        # SUPABASE INIT (Cloud Memory)
        url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if url and supabase_key:
            self.supabase: Client = create_client(url, supabase_key)
        else:
            self.supabase = None
            logger.warning("Supabase keys missing. Learning without cloud persistence.")

       #micro-batch setup
        self._click_counter = 0
        self.BATCH_SIZE = 5 # 5 for faster local testing 
    


    def download_latest_brain(self) -> bool:
        """Downloads all required .pkl and .sig artifacts from Supabase on startup"""
        if not self.supabase:
            logger.warning("Supabase not configured skipping cloud model download")
            return False
        
        base_path = Path(__file__).resolve().parent.parent.parent.parent
        artifacts_dir = base_path / "zone_2_artifacts"
        
        # List of all files the server needs to be 'whole'
        artifacts = [
            "bandit_model.pkl", 
            "bandit_model.pkl.sig",
            "cmab_preprocessor.pkl",
            "cmab_preprocessor.pkl.sig"
        ]

        try:
            logger.info(" Synchronizing Hybrid AI artifacts from Supabase...")
            
            for filename in artifacts:
                file_path = artifacts_dir / filename
                
                # Download from 'ai-models' bucket
                model_bytes = self.supabase.storage.from_("ai-models").download(filename)
                
                with open(file_path, "wb") as f:
                    f.write(model_bytes)
                
                logger.info(f"   Successfully synced: {filename}")
                
            logger.info(" Cloud synchronization complete. Ready to load resources.")
            return True
            
        except Exception as e:
            logger.warning(f"Cloud sync failed (Falling back to local artifacts): {e}")
            return False
        




    def _verify_signature(self, file_path: Path) -> bool:
        """
        Verifies that the file matches its .sig signature.
        Prevents loading malicious pickles.
        Reads file in 1MB chucks to prevent RAM spikes.
        """
        sig_path = file_path.with_suffix(file_path.suffix + ".sig")
        if not sig_path.exists():
            logger.error(f" SECURITY FAIL: Missing signature for {file_path.name}")
            return False

        # 1. Calculate current hash
        try:
            # 1. initialize HMAC (streaming MOde)
            h = hmac.new(self.secret_key, digestmod=hashlib.sha256)
            # 2. Read file in chunks
            CHUNK_SIZE = 1024 * 1024 # only 1MB
            with open(file_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    h.update(chunk)
            calculated_sig = h.hexdigest()

            # 3. Read stored hash
            with open(sig_path, "r") as f:
                stored_sig = f.read().strip()

            # 3. Compare safely (prevent timing attacks)
            if hmac.compare_digest(calculated_sig, stored_sig):
                return True
            else:
                logger.error(f" SECURITY FAIL: Signature mismatch for {file_path.name}")
                return False
        except Exception as e:
            logger.error(f" SECURITY ERROR during verification: {e}")
            return False
        

    def _sign_model(self, file_path: Path)->None:
        """Automatically sign the model files , keeping the services self contained"""
        sig_path = file_path.with_suffix(file_path.suffix + ".sig")
        h = hmac.new(self.secret_key, digestmod= hashlib.sha256)

        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        with open(sig_path, "w") as f:
            f.write(h.hexdigest())


    def load_resources(self) -> bool:
        """
        Loads ML artifacts. Returns True if successful.
        """
        logger.info(" PredictionService: Loading artifacts...")
        
        # Robust path finding (Goes up from: services -> app -> zone_3 -> root)
        base_path = Path(__file__).resolve().parent.parent.parent.parent
        artifacts_dir = base_path / "zone_2_artifacts"

        try:
            # 1. Load Bandit (With Security Check)
            bandit_path = artifacts_dir / "bandit_model.pkl"
            if not self._verify_signature(bandit_path):
                raise ModelIntegrityError(f"Integrity check failed for {bandit_path}")
            
            with open(bandit_path, "rb") as f:
                self.bandit = pickle.load(f)
                
            #THE HYBRID SHIFT: Switch from Offline (1.0) to Live Online Mode (0.90)
            self.bandit.decay_factor = 0.90
            logger.info(" Shifted Bandit to Live Online Mode (Decay: 0.90)")
            
            # 2. Load Preprocessor (With Security Check)
            prep_path = artifacts_dir / "cmab_preprocessor.pkl"
            if not self._verify_signature(prep_path):
                raise ModelIntegrityError(f"Integrity check failed for {prep_path}")

            with open(prep_path, "rb") as f:
                self.preprocessor = pickle.load(f)

            # 3. load config (sync Actions with Trainer)
            config_path = artifacts_dir / "training_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    # covert keys to strings to ints ( json stores keys as strings)
                    self.actions = {int(k): v for k, v in config.get("actions",{}).items()}
                logger.info(f"Loaded {len(self.actions)} actions from config.")
            else:
                logger.warning("Config not found using fallback actions.")
                self.actions = {0: "strict_budget", 1: "streak_builder", 2: "quiz", 3: "cool_off"}
            # connect LLM
            self.advisor = LLMStrategyAdvisor(skip_api_init=False)
            
            self._is_ready = True
            logger.info(" Hybrid Brain Loaded & Verified Successfully!")
            return True

        except ModelIntegrityError as e:
            logger.critical(f" CRITICAL SECURITY ALERT: {e}")
            return False
        except Exception as e:
            logger.critical(f" FATAL ERROR loading resources: {e}")
            return False

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Async prediction pipeline using Pydantic for type safety and Parallel Execution.
        """
        if not self._is_ready:
            raise RuntimeError("Service not initialized. Models are not loaded.")
        
        pred_id = str(uuid.uuid4())

        try:
            # SUPABASE CLOUD SYNC ---
            expense_id = None
            if self.supabase:
                try:
                    # 1. Ensure the User exists
                    self.supabase.table("users").upsert({
                        "user_id": request.user_id,
                        "test_group": request.test_group
                    }).execute()

                    # 2. Look up the category_id
                    cat_res = self.supabase.table("categories").select("category_id").eq("name", request.category).execute()
                    category_id = cat_res.data[0]["category_id"] if cat_res.data else 1 

                    # 3. Save the Expense
                    exp_res = self.supabase.table("expenses").insert({
                        "user_id": request.user_id,
                        "category_id": category_id,
                        "amount": request.amount
                    }).execute()
                    
                    expense_id = exp_res.data[0]["expense_id"]
                except Exception as db_err:
                    logger.error(f"Supabase Cloud Sync Error: {db_err}")
            # -----------------------------------

            # 1. Preprocess (uses helper for cleaner and type-safe)
            context_df = request.to_dataframe()
            context_vector = self.preprocessor.transform(context_df)

            # 2. Bandit Decision (Fast - CPU bound)
            chosen_arm_idx, debug_info = self.bandit.select_arm(context_vector[0])
            native_arm_idx = int(chosen_arm_idx)
            action_name = self.actions.get(native_arm_idx, "unknown_action")

            # 3. LLM Enhancement (parallel execution safely awaited)
            features_dict = request.features.model_dump()
            
            strategy, notification = await asyncio.gather(
                self.advisor.get_strategy_recommendation(features_dict),
                self.advisor.generate_message(features_dict, action_name)
            )

            # 4. Save context state for feedback loop
            arm_debug = debug_info.get(chosen_arm_idx, {})
            uncertainty_score = float(arm_debug.get("uncertainty", 0.0))
            user_volatility = getattr(request.features, 'spending_volatility', 0.0)
            segment = "High Volatility" if user_volatility > 0.8 else "Stable"

            cache_payload = json.dumps({
                "arm_index": native_arm_idx,
                "context": context_vector[0].tolist(),
                "uncertainty": uncertainty_score, 
                "segment": segment,
                "user_id": request.user_id,
                "expense_id": expense_id #Remember the expense for the feedback loop
            })
            await self.advisor.cache.put(f"pred_{pred_id}", cache_payload)

            # Return Pydantic Response
            return PredictionResponse(
                prediction_id=pred_id,
                user_id=request.user_id,
                strategy=strategy,
                action=action_name,
                notification=notification,
                visual_theme="red" if action_name == "strict_budget" else "blue",
                debug_info=debug_info.get(chosen_arm_idx, {})
            )

        except Exception as e:
            logger.error(f" Prediction Error: {e}")
            # safe Fallback (prevents app crash)
            return PredictionResponse(
                prediction_id=pred_id,
                user_id=request.user_id,
                strategy="ERROR",
                action="cool_off",
                notification="Let's take a moment to reflect on your goals.",
                visual_theme="gray",
                debug_info={"error": str(e)}
            )
        

    async def process_feedback(self, request: FeedbackRequest) -> None:
        """Background task: Fetches cached context, updates Bandit, saves securely to disk"""
        try:
            # 1. Fetch exact state from cache
            cache_key = f"pred_{request.prediction_id}"
            cached_data_str = await self.advisor.cache.get(cache_key)

            if not cached_data_str:
                logger.warning(f"Feedback dropped: prediction_id {request.prediction_id} expired or invalid.")
                return
            #Instantly delete the data from memory to prevent RAM leaks!
            try:
                
                await self.advisor.cache.remove(cache_key) 
            except Exception as e:
                logger.warning(f"Could not clear cache for {cache_key}: {e}")
            #2 Reconstruct math

            cached_data = json.loads(cached_data_str)
            arm_index = cached_data["arm_index"]
            context_vector = np.array(cached_data["context"])
            uncertainty_score = cached_data.get("uncertainty", 0.0) #read from cache
            segment = cached_data.get("segment", "Unknown")

            # 3 update the bandit brain
            self.bandit.update(arm_index,context_vector, request.reward)

            #4 Crash safe save and re sign internally
            base_path = Path(__file__).resolve().parent.parent.parent.parent
            model_path = base_path / "zone_2_artifacts" / "bandit_model.pkl"
            temp_path = model_path.with_suffix('.pkl.tmp')

            #  write to temporary file first
            with open (temp_path, "wb") as f:
                pickle.dump(self.bandit, f)

            # Rename  ( atomic operation on most os, prevents corruption)
            temp_path.replace(model_path)
            # clean, internal re-signing
            self._sign_model(model_path)

            
            #5. Log to Supabase PostgreSQL (Permanent Analytics)
           
            action_name = self.actions.get(arm_index, "unknown")  # Define action_name first by mapping the arm_index
            user_id = cached_data.get("user_id") #Retrieve from cache
            expense_id = cached_data.get("expense_id")
            if self.supabase:
                try:
                    self.supabase.table("analytics_log").insert({
                        "prediction_id": request.prediction_id,
                        "user_id": user_id,
                        "expense_id": expense_id,
                        "action_name": action_name,
                        "reward": request.reward,
                        "uncertainty": uncertainty_score, #save to cloud
                        "user_segment": segment
                    }).execute()
                except Exception as db_err:
                    logger.error(f"Supabase Analytics Error: {db_err}")

            # 6.Micro-Batching to Supabase Object Storage
            self._click_counter += 1
            if self._click_counter >= self.BATCH_SIZE:
                if self.supabase:
                    try:
                        logger.info(f" Batch limit reached ({self.BATCH_SIZE}). Uploading model to Supabase...")
                        
                        # Upload the Model (.pkl)
                        with open(model_path, 'rb') as f:
                            self.supabase.storage.from_("ai-models").upload(
                                file=f, 
                                path="bandit_model.pkl", 
                                file_options={"upsert": "true"}
                            )
                            
                        # Upload the Signature (.pkl.sig)
                        sig_path = model_path.with_suffix('.pkl.sig')
                        with open(sig_path, 'rb') as f:
                            self.supabase.storage.from_("ai-models").upload(
                                file=f, 
                                path="bandit_model.pkl.sig", 
                                file_options={"upsert": "true"}
                            )
                        logger.info(" Cloud persistence successful.")
                    except Exception as upload_err:
                        logger.error(f"Supabase Storage Error: {upload_err}")
                
                # Reset counter regardless of success to prevent spamming
                self._click_counter = 0
            logger.info(f"AI Learned! Reward: {request.reward} for arm: {arm_index} (Pred: {request.prediction_id})")
        except Exception as e:
            logger.error(f"Failed to process background feedback: {e}", exc_info=True)

# Singleton Pattern
prediction_service = PredictionService()