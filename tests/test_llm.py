import sys
import os
import time
import asyncio  # <--- NEW: Needed for async functions
from pathlib import Path
from dotenv import load_dotenv  

# 1. Setup Paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 2. Load Environment Variables 
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# 3. Import the Brain
try:
    from shared_core.llm_advisor import LLMStrategyAdvisor
    print(" Successfully imported LLMStrategyAdvisor")
except ImportError as e:
    print(f" Import Error: {e}")
    sys.exit(1)

# NEW: Must be 'async' to use 'await'
async def test_live_connection():
    print("\n TESTING GEMINI API CONNECTION...")
    
    # Check if key loaded
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(" CRITICAL: GEMINI_API_KEY not found in .env file.")
        return

    print(" API Key loaded successfully.")

    # Initialize Advisor
    advisor = LLMStrategyAdvisor(skip_api_init=False)
    
    # 1. Test Strategy Decision
    test_user = {'spending_volatility': 0.95, 'return_rate': 0.05}
    print(f"\n1. Sending Test Context: {test_user}")
    
    start = time.time()
    
    # NEW: Added 'await'
    strategy = await advisor.get_strategy_recommendation(test_user)
    
    duration = time.time() - start
    
    print(f"    Decision: {strategy}")
    print(f"    Time: {duration:.2f}s")
    
    if strategy == "EXPLORE":
        print("    SUCCESS: Logic matches (High Volatility -> Explore)")
    else:
        print(f"   WARNING: Received {strategy} (Expected EXPLORE)")

    # 2. Test Message Generation
    print("\n2. Testing Message Generation...")
    
    # NEW: Added 'await'
    msg = await advisor.generate_message(test_user, 0, "strict_budget")
    
    print(f"    Notification: \"{msg}\"")
    print("\n   TEST COMPLETE: If you see a notification above, Gemini is working!")

if __name__ == "__main__":
    # NEW: Run the async loop
    asyncio.run(test_live_connection())