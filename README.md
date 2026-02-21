Here is the complete, finalized `README.md` with all the advanced technical details included. You can copy the entire block below and paste it directly into your file.

```markdown
# 🧠 Adaptive Finance AI (Hybrid Bandit + LLM)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Adaptive Finance AI** is a high-performance, privacy-focused financial recommendation engine built for mobile applications. 

It utilizes a **Hybrid AI Architecture** combining:
1.  **Contextual Bandits (LinUCB):** A mathematical model that learns optimal financial strategies (e.g., *Strict Budget*, *Cool Off*, *Quiz*) based on user spending volatility and engagement.
2.  **Generative AI (Google Gemini 2.5):** A creative layer that translates raw strategies into witty, personalized, and empathetic user notifications.

This backend serves as the stateless "Brain" for the Adaptive Finance Android App, continuously learning from user feedback in real-time.

---

## 🚀 Key Features

* **⚡ Non-Blocking I/O (Async/Await):** Fully asynchronous architecture using FastAPI. LLM generation and Bandit math run in parallel (`asyncio.gather`), ensuring sub-second inference.
* **🎯 Online Contextual Learning:** Implements `LinUCB` with Sherman-Morrison updates for O(d²) online learning efficiency. Features a FastAPI `BackgroundTasks` queue so the Bandit can update its matrix without lagging the mobile UI.
* **💾 Stateless Caching Architecture:** Uses an asynchronous, WAL-mode SQLite cache (`aiosqlite`) to securely store prediction states. This prevents "Ghost Coroutines" and keeps the Android client lightweight.
* **🛡️ Cryptographic Security:**
    * **Model Integrity:** All AI models (`.pkl`) are digitally signed with HMAC-SHA256. The system performs an atomic file replacement (`os.replace`) and re-signs the model dynamically upon learning.
    * **API Security:** Endpoints are protected via a custom `X-API-Token` header and strict `.env` driven CORS policies.
* **💬 Generative Persona:** Uses Google Gemini 2.5 to generate dynamic content. Implements a custom Async Circuit Breaker to instantly fallback to hardcoded strings if the LLM API spikes in latency.

---

## 🏗️ File Structure

The project follows a strict **Micro-Zoned Architecture** to separate Data Science logic from Web Backend logic.

```text
adaptive_finance_ai/
├── scripts/
│   └── sign_models.py           # Cryptographically signs trained models (.pkl) with HMAC
├── shared_core/                 # Shared resources between Training and Inference
│   ├── features.py              # Centralized feature engineering and definitions
│   ├── llm_advisor.py           # Async Google Gemini integration, Circuit Breaker & aiosqlite Cache
│   ├── models.py                # Custom LinUCB Contextual Bandit implementation
│   ├── preprocessing.py         # KBinsDiscretizer & Log1p math pipelines
│   └── schemas.py               # Strict Pydantic contracts for API validation
├── zone_1_training/             # Offline Data Science & Machine Learning
│   ├── src/
│   │   ├── cmab_env.py          # Simulated environment for reinforcement learning
│   │   └── cmab_visualizations.py # Generates the 9-panel object-oriented dashboard
│   ├── synthetic_data.py        # Generates Kenyan market user data for pre-training
│   └── trainer.py               # Main training script (Teacher Forcing & Epsilon Decay)
├── zone_2_artifacts/            # The "Vault" (Secure Bridge between Zones 1 and 3)
│   ├── bandit_model.pkl         # The trained LinUCB brain (Binary)
│   ├── bandit_model.pkl.sig     # The HMAC SHA-256 signature for the bandit
│   ├── cmab_preprocessor.pkl    # The trained feature scaler/binner (Binary)
│   ├── cmab_preprocessor.pkl.sig# The HMAC SHA-256 signature for the preprocessor
│   ├── training_config.json     # Action map and training metadata
│   └── cmab_comprehensive_analysis.png # Visual proof of AI learning convergence
├── zone_3_inference/            # Production Web Backend
│   └── app/
│       ├── main.py              # FastAPI Gateway (CORS, Background Tasks, Endpoints)
│       └── services/
│           └── prediction_service.py # Core logic bridging the Bandit, LLM, and UUID Cache
├── .env                         # Environment variables (Ignored in Git)
├── .gitignore                   # Prevents secret keys and SQLite databases from leaking
└── requirements.txt             # Python dependencies (aiosqlite, fastapi, google-genai, etc.)

```

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.10+
* Google AI Studio API Key (for Gemini)

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/adaptive-finance-backend.git](https://github.com/YOUR_USERNAME/adaptive-finance-backend.git)
cd adaptive-finance-backend

# Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
pip install -e .           # Install local packages for shared_core routing

```

### 2. Configuration (`.env`)

Create a `.env` file in the root directory:

```ini
# --- AI Configuration ---
GEMINI_API_KEY=your_google_gemini_key_here

# --- Security ---
# The password your App sends to the Server
API_ACCESS_TOKEN=my_thesis_access_token_2026

# Key used to sign/verify model integrity
MODEL_SIGNING_KEY=my_super_secret_thesis_key_2026

# --- CORS Rules ---
# Comma separated list. 10.0.2.2 is required for Android Emulator testing.
ALLOWED_ORIGINS=http://localhost:8000,[http://10.0.2.2:8000](http://10.0.2.2:8000)

```

### 3. Run Locally

```bash
python zone_3_inference/app/main.py

```

Visit `http://127.0.0.1:8000/docs` to access the Swagger UI and test the endpoints.

---

## 📡 API Usage (The Stateless Loop)

The system operates on a two-step reinforcement learning loop using background tasks and an async SQLite cache.

### Step 1: Request Prediction

**Endpoint:** `POST /predict/v1/context`

**Headers:** `Content-Type: application/json` | `X-API-Token: <YOUR_ACCESS_TOKEN>`

**Request:**

```json
{
  "user_id": "user_123",
  "features": {
    "total_spend": 1200.0,
    "spending_volatility": 0.95,
    "return_rate": 0.05,
    "transaction_count": 50,
    "avg_transaction_value": 20.0
  }
}

```

**Response:**

```json
{
  "prediction_id": "e0c9ed02-d764-40e9-b85f-43bb354e104b",
  "user_id": "user_123",
  "strategy": "EXPLORE",
  "action": "strict_budget",
  "notification": "Whoa! Your spending is jumping around excessively! 🚨 Activating strict budget mode.",
  "visual_theme": "red",
  "debug_info": {}
}

```

### Step 2: Submit Feedback (Learning)

*The Android app must send back the `prediction_id` so the AI can learn from the interaction.*

**Endpoint:** `POST /predict/v1/feedback`

**Headers:** `Content-Type: application/json` | `X-API-Token: <YOUR_ACCESS_TOKEN>`

**Request:**

```json
{
  "prediction_id": "e0c9ed02-d764-40e9-b85f-43bb354e104b",
  "reward": 1.0 
}

```

*(Reward is `1.0` if the user clicked/engaged, `0.0` if they ignored the notification).*

**Response:** Returns a `202 Accepted` status immediately, while the Bandit matrix updates and atomically re-signs its `.pkl` file in a background worker thread.

---

## 🧪 Testing & Training

**Retraining the Brain from Scratch:**
To rebuild the Data Science environment, test Epsilon Decay, and generate the 9-panel dashboard:

```bash
python zone_1_training/trainer.py

```

**Manual Security Verification:**
To verify the cryptographic integrity of the models without booting the server:

```bash
python scripts/sign_models.py

```

---

📜 **License:** Distributed under the MIT License.

🎓 **Developed for:** BSc Computer Science Final Year Project (2026).

```

```