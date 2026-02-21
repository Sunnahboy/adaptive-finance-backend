
import pickle
import asyncio
import hmac
import hashlib
import os
import logging
import json
import uuid
import numpy as np


import pandas as pd  
from pathlib import Path
from typing import Optional, Dict
from fastapi.concurrency import run_in_threadpool

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
        
        # Action Map (Must match Trainer) load from training_config.json
        self.actions : Dict[int,str] = {}
        self._is_ready = False
        
        # SECURITY: Load key from env 
        key = os.getenv("MODEL_SIGNING_KEY")
        if not key:
            raise RuntimeError ("CRITICAL: MODEL_SIGNING_KEY not set! Server refuses to start.")
        self.secret_key = key.encode()
        

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
            #1. Preprocess (uses helper for cleaner and type-safe)
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

            #4. save context state for feedback loop
            cache_payload = json.dumps({
                "arm_index": native_arm_idx,
                "context": context_vector[0].tolist()
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
            #2 Reconstruct math
            cached_data = json.loads(cached_data_str)
            arm_index = cached_data["arm_index"]
            context_vector = np.array(cached_data["context"])
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
            temp_path.rename(model_path)
            # clean, internal re-signing
            self._sign_model(model_path)
            logger.info(f"AI Learned! Reward: {request.reward} for arm: {arm_index} (Pred: {request.prediction_id})")
        except Exception as e:
            logger.error(f"Failed to process background feedback: {e}", exc_info=True)

# Singleton Pattern
prediction_service = PredictionService()