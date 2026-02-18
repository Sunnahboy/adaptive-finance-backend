
import sys
import time
import pickle
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== REPRODUCIBILITY =====
# MUST be set BEFORE any numpy/sklearn operations
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

np.random.seed(42)

# ===== PATH SETUP =====
CURRENT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared_core.features import calculate_user_features
    from shared_core.models import LinUCB
    from shared_core.preprocessing import BanditPreprocessor, BanditPreprocessorConfig
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)

# Visualization
try:
    from zone_1_training.src.cmab_visualizations import plot_comprehensive_training_analysis
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# ===== CONFIGURATION =====
ACTIONS: Dict[int, str] = {
    0: "strict_budget",
    1: "streak_builder",
    2: "quiz",
    3: "cool_off",
}

#prevents logic errors in case of actions are reordered
ACTION_MAP =  {v: k for k, v in ACTIONS.items()}
# Dynamic priority
PRIORITY_ORDER: List[int] = [
    ACTION_MAP["cool_off"],
    ACTION_MAP["strict_budget"],
    ACTION_MAP["quiz"],
    ACTION_MAP["streak_builder"]
]

REWARD_CONFIG = {
    "volatility_threshold": 0.8,
    "return_threshold": 0.15,
    "low_engagement_threshold": 10,
    "high_value_threshold": 50,
}

# hyperparameters 
TRAINING_CONFIG = {
    "alpha": 0.5,                # High exploration for bootstrapping (prevent overconfidence)
    "allow_diagonal": False,     # Full covariance matrix for accuracy
    "decay_factor": 1.0,         
    "decay_strategy": "interpolation",
    "use_teacher_forcing": True, # Critical for learning
}

CONTEXT_NUMERIC_COLS = [
    "spending_volatility",
    "return_rate",
    "transaction_count",
    "avg_transaction_value",
]
CONTEXT_CATEGORICAL_COLS: Sequence[str] = []

# ===== REWARD LOGIC =====
def get_synthetic_reward(feats: Dict, arm_idx: int) -> float:
    """Calculates ground-truth reward based on business rules."""
    vol = feats.get("spending_volatility", 0.0)
    ret = feats.get("return_rate", 0.0)
    count = feats.get("transaction_count", 0.0)
    avg_txn = feats.get("avg_transaction_value", 0.0)

    # High returns  cool off
    if arm_idx ==  ACTION_MAP["cool_off"] and ret > REWARD_CONFIG["return_threshold"]: 
        return 1.0
    # High volatility strict budget 
    if arm_idx == ACTION_MAP["strict_budget"] and vol > REWARD_CONFIG["volatility_threshold"]: 
        return 1.0
    #moderate returns or high spender quiz (educational)
    if arm_idx == ACTION_MAP["quiz"] and (0.05 < ret <= REWARD_CONFIG["return_threshold"] or avg_txn > REWARD_CONFIG["high_value_threshold"]): 
        return 1.0
    #Low activity streak builder ( Gamification)
    if arm_idx == ACTION_MAP["streak_builder"] and 0 < count < REWARD_CONFIG["low_engagement_threshold"]: 
        return 1.0
    
    return 0.0

def find_best_arm_by_priority(feats: Dict) -> Tuple[Optional[int], float]:
    """Finds optimal arm for teacher forcing."""
    for arm in PRIORITY_ORDER:
        if get_synthetic_reward(feats, arm) > 0.0:
            return arm, 1.0
    return None, 0.0

def pre_calculate_strategy_labels(df: pd.DataFrame) -> Dict[int, str]:
    """Vectorized Explore/Exploit labeling."""
    vol_thresh = REWARD_CONFIG["volatility_threshold"]
    ret_thresh = REWARD_CONFIG["return_threshold"]

    vols = df["spending_volatility"].values
    rets = df["return_rate"].values
    indices = df.index.values
     # Vectorized check : if user is risky (high vol or high returns) we enforce exploration
    labels = np.where((vols > vol_thresh) | (rets > ret_thresh), "EXPLORE", "EXPLOIT")
    return dict(zip(indices, labels))

# ===== MAIN TRAINING LOOP =====
def train_and_validate() -> None:
    print("=" * 70)
    print("ADAPTIVE FINANCE AI - ROBUST TRAINING PIPELINE")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")
    print(f"Random state: {np.random.randint(0, 1000)}")
    print()

    start_time = time.time()
    
    # Setup Artifacts
    artifacts_dir = PROJECT_ROOT / "zone_2_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f" Artifacts: {artifacts_dir}")

    # 1. Load Data
    data_path = PROJECT_ROOT / "zone_1_training" / "data" / "interim" / "cleaned_transactions.csv"
    if not data_path.exists():
        data_path = Path("data/interim/cleaned_transactions.csv")
    
    if not data_path.exists():
        print(f" Data not found at {data_path}")
        return

    print(" Loading and preprocessing data...")
    raw_df = pd.read_csv(data_path)
    full_df = calculate_user_features(raw_df)
    print(f"   Loaded {len(full_df)} user records")

    # 2. Split Data
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. Feature Processing
    print(" Fitting preprocessor...")
    pre_cfg = BanditPreprocessorConfig(
        numeric_cols=CONTEXT_NUMERIC_COLS,
        categorical_cols=CONTEXT_CATEGORICAL_COLS,
    )
    preprocessor = BanditPreprocessor(pre_cfg)
    
    train_matrix = preprocessor.fit_transform(train_df[CONTEXT_NUMERIC_COLS + list(CONTEXT_CATEGORICAL_COLS)])
    test_matrix = preprocessor.transform(test_df[CONTEXT_NUMERIC_COLS + list(CONTEXT_CATEGORICAL_COLS)])
    
    print(f"   Feature matrix shape: {train_matrix.shape}")

    # Save preprocessor
    with open(artifacts_dir / "cmab_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # 4. Initialize Bandit
    print(" Initializing LinUCB bandit...")
    bandit = LinUCB(
        n_arms=len(ACTIONS),
        n_features=train_matrix.shape[1],
        alpha=TRAINING_CONFIG["alpha"],
        allow_diagonal=TRAINING_CONFIG["allow_diagonal"],
        decay_factor=TRAINING_CONFIG["decay_factor"],
        decay_strategy=TRAINING_CONFIG["decay_strategy"],
    )

    # 5. Prepare Training
    print(" Distilling strategy knowledge...")
    strategy_labels = pre_calculate_strategy_labels(train_df)
    
    train_records = train_df.to_dict("records")
    train_indices = train_df.index.tolist()

    history_arms: List[int] = []
    history_rewards: List[float] = []
    history_arm_selections: Dict[int, List[int]] = {k: [] for k in ACTIONS}
    history_arm_rewards: Dict[int, List[float]] = {k: [] for k in ACTIONS}

    # ===== TRAINING LOOP =====
    print(f"\n[TRAINING] Processing {len(train_matrix)} users...")
    
    teacher_force_count = 0
    explore_count = 0

    for i, ctx_vec in enumerate(train_matrix):
        if i % 500 == 0 and i > 0:
            print(f"    Processed {i:,} / {len(train_matrix):,} users...")

        feats = train_records[i]
        idx = train_indices[i]

        strategy = strategy_labels.get(idx, "EXPLOIT")

        if strategy == "EXPLORE":
            chosen_arm = np.random.randint(0, len(ACTIONS))
            explore_count += 1
        else:
            chosen_arm, _ = bandit.select_arm(ctx_vec)

        reward = get_synthetic_reward(feats, chosen_arm)

        # SUPERIOR teacher forcing logic
        # Key difference: Only teach during EXPLOIT, not EXPLORE
        if reward > 0.0:
            # if the bandit got it right , reinforce it
            bandit.update(chosen_arm, ctx_vec, reward)
        else:
            # if wrong, only correct it during the exploit phase
            if strategy == "EXPLOIT": 
                #1. punish the wrong choice
                bandit.update(chosen_arm, ctx_vec, 0.0)
                #2. Teach the correct choice (oracle update)
                if TRAINING_CONFIG["use_teacher_forcing"]:
                    best_arm, best_r = find_best_arm_by_priority(feats)
                    if best_arm is not None and best_arm != chosen_arm:
                        bandit.update(best_arm, ctx_vec, best_r)
                        teacher_force_count += 1

        history_arms.append(chosen_arm)
        history_rewards.append(reward)
        history_arm_selections[chosen_arm].append(i)
        history_arm_rewards[chosen_arm].append(reward)

    print(f"\n    Training complete!")
    print(f"   - Exploration episodes: {explore_count:,}")
    print(f"   - Teacher forcing corrections: {teacher_force_count:,}")

    # 6. Visualization
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating analysis plots...")
        try:
            plot_comprehensive_training_analysis(
                arm_history=history_arms,
                reward_history=history_rewards,
                arm_selection_history=history_arm_selections,
                arm_reward_history=history_arm_rewards,
                output_dir=str(artifacts_dir),
            )
            print(f"    Graphs saved")
        except Exception as e:
            print(f"    Visualization failed: {e}")

    # 7. Validation
    print("\n[VALIDATION]")
    test_records = test_df.to_dict("records")
    correct = 0
    total_regret = 0.0
    
    for i, ctx_vec in enumerate(test_matrix):
        feats = test_records[i]
        pred_arm, _ = bandit.select_arm(ctx_vec)

        r = get_synthetic_reward(feats, pred_arm)
        
        # Optimal reward (oracle knowledge)
        opt_r = 0.0
        for arm in range(len(ACTIONS)):
            if get_synthetic_reward(feats, arm) > 0.0:
                opt_r = 1.0
                break

        if r > 0.0:
            correct += 1
        total_regret += (opt_r - r)

    accuracy = correct / len(test_matrix)
    avg_regret = total_regret / len(test_matrix)

    elapsed = time.time() - start_time

    print(f"    Accuracy:    {accuracy:.2%}")
    print(f"    Avg Regret:  {avg_regret:.4f}")
    print(f"    Total Time:   {elapsed:.2f}s")

    # 8. Save Model & Config
    with open(artifacts_dir / "bandit_model.pkl", "wb") as f:
        pickle.dump(bandit, f)
    
    config_data = {
        "model_type": "LinUCB",
        "actions": ACTIONS,
        "action_map": ACTION_MAP, # save for reference
        "reward_config": REWARD_CONFIG,
        "training_config": TRAINING_CONFIG,
        "metrics": {
            "accuracy": float(accuracy),
            "avg_regret": float(avg_regret),
            "training_samples": len(train_matrix),
            "test_samples": len(test_matrix),
            "training_time_seconds": elapsed,
            "timestamp": time.time()
        }
    }
    
    with open(artifacts_dir / "training_config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n Training complete! Artifacts saved to {artifacts_dir}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        train_and_validate()
    except KeyboardInterrupt:
        print("\n Training interrupted by user.")
    except Exception as e:
        print(f"\n FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()



