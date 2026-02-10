
import pickle
import asyncio
import hmac
import hashlib
import os
import logging
import pandas as pd  
from pathlib import Path
from typing import Optional
from fastapi.concurrency import run_in_threadpool

# CLEAN IMPORTS (Assumes 'pip install -e .' was run)
try:
    from shared_core.llm_advisor import LLMStrategyAdvisor
    from shared_core.models import LinUCB
    from shared_core.preprocessing import BanditPreprocessor
    from shared_core.schemas import PredictionRequest, PredictionResponse
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
        
        # Action Map (Must match Trainer)
        self.actions = {0: "strict_budget", 1: "streak_builder", 2: "quiz", 3: "cool_off"}
        self._is_ready = False
        
        # SECURITY: Load key from env (Default provided for dev, override in prod!)
        self.secret_key = os.getenv("MODEL_SIGNING_KEY", "my_super_secret_thesis_key_2026").encode()

    def _verify_signature(self, file_path: Path) -> bool:
        """
        Verifies that the file matches its .sig signature.
        Prevents loading malicious pickles.
        """
        sig_path = file_path.with_suffix(file_path.suffix + ".sig")
        if not sig_path.exists():
            logger.error(f" SECURITY FAIL: Missing signature for {file_path.name}")
            return False

        # 1. Calculate current hash
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            calculated_sig = hmac.new(self.secret_key, file_data, hashlib.sha256).hexdigest()

            # 2. Read stored hash
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

            # 3. Connect LLM
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

        try:
            # 1. Preprocess (Fast - CPU bound)
            features_dict = request.features.model_dump()
    
            df = pd.DataFrame([features_dict])
            context_vector = self.preprocessor.transform(df)

            # 2. Bandit Decision (Fast - CPU bound)
            chosen_arm_idx, debug_info = self.bandit.select_arm(context_vector[0])
            action_name = self.actions[chosen_arm_idx]

            # 3. LLM Enhancement (Slow - I/O bound)
            # Call async functions directly (No run_in_threadpool needed)
            
            # Fire off both tasks simultaneously
            strategy_task = self.advisor.get_strategy_recommendation(features_dict)
            notification_task = self.advisor.generate_message(
                features_dict, 
                chosen_arm_idx, 
                action_name
            )

            # Wait for BOTH to finish (latency is max(T1, T2))
            # The event loop handles the concurrency efficiently
            strategy, notification = await asyncio.gather(strategy_task, notification_task)

            # Return Pydantic Response
            return PredictionResponse(
                user_id=request.user_id,
                strategy=strategy,
                action=action_name,
                notification=notification,
                visual_theme="red" if action_name == "strict_budget" else "blue",
                debug_info=debug_info[chosen_arm_idx]
            )

        except Exception as e:
            logger.error(f" Prediction Error: {e}")
            raise e

# Singleton Pattern
prediction_service = PredictionService()