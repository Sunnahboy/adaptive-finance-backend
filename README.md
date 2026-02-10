# 🧠 Adaptive Finance AI (Hybrid Bandit + LLM)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Adaptive Finance AI** is a high-performance, privacy-focused financial recommendation engine built for mobile applications. 

It utilizes a **Hybrid AI Architecture** combining:
1.  **Contextual Bandits (LinUCB):** A mathematical model that learns optimal financial strategies (e.g., *Strict Budget*, *Cool Off*, *Quiz*) based on user spending volatility and engagement.
2.  **Generative AI (Google Gemini 2.5):** A creative layer that translates raw strategies into witty, personalized, and empathetic user notifications.

This backend serves as the "Brain" for the Adaptive Finance Android App.

---

## 🚀 Key Features

* **⚡ Real-Time Inference:** Sub-second strategy prediction using `FastAPI` and asynchronous processing.
* **🎯 Contextual Learning:** Implements `LinUCB` with Sherman-Morrison updates for O(d²) online learning efficiency.
* **🛡️ Cryptographic Security:**
    * **Model Integrity:** All AI models (`.pkl`) are digitally signed with HMAC-SHA256 to prevent tampering.
    * **API Security:** Endpoints are protected via a custom `X-API-Token` header.
* **💬 Generative Persona:** Uses Google Gemini to generate dynamic content, avoiding repetitive "bot-like" notifications.
* **📊 Automated Training Pipeline:** Includes a dedicated training zone that generates comprehensive performance visualizations automatically.

---

## 🏗️ System Architecture

The project follows a strict **Micro-Zoned Architecture**:

* **Zone 1 (Training):**
    * Ingests raw transaction data.
    * Trains the `LinUCB` bandit.
    * Generates performance graphs (`cmab_comprehensive_analysis.png`).
* **Zone 2 (Artifacts - The "Vault"):**
    * Stores the "frozen" intelligence (Signed Models & Scalers).
    * Acts as the secure bridge between Training and Inference.
* **Zone 3 (Inference - The "API"):**
    * A lightweight `FastAPI` service.
    * Loads artifacts securely on startup.
    * Exposes REST endpoints for the mobile app.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.10+
* Docker (Optional, for containerization)
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


2. Configuration (.env)
Create a .env file in the root directory:

Ini, TOML
# --- AI Configuration ---
GEMINI_API_KEY=your_google_gemini_key_here

# --- Security ---
# The password your App sends to the Server
API_ACCESS_TOKEN=my_thesis_access_token_2026

# Key used to sign/verify model integrity
MODEL_SIGNING_KEY=my_super_secret_thesis_key_2026

# --- System ---
LOG_LEVEL=info
3. Run Locally
Bash
uvicorn zone_3_inference.app.main:app --host 0.0.0.0 --port 8000 --reload
Visit https://www.google.com/search?q=http://127.0.0.1:8000/docs to access the Swagger UI.

🐳 Docker Deployment
This project is fully containerized and ready for cloud deployment (Render, AWS, Fly.io).

Bash
# Build the image
docker build -t adaptive-finance-api .

# Run the container
docker run -p 8000:8000 --env-file .env adaptive-finance-api
📡 API Usage
Endpoint: POST /predict/v1/context

Headers:

Content-Type: application/json

X-API-Token: <YOUR_ACCESS_TOKEN>

Request Body:

JSON
{
  "user_id": "user_123",
  "features": {
    "spending_volatility": 0.95,
    "return_rate": 0.05,
    "transaction_count": 50,
    "avg_transaction_value": 20.0,
    "total_spend": 1200.0
  }
}
Response:

JSON
{
  "strategy": "strict_budget",
  "confidence": 0.95,
  "message": "Whoa! Your spending is jumping around excessively! 🚨 Activating strict budget mode.",
  "meta": {
    "version": "1.0.0",
    "model": "LinUCB + Gemini 2.5"
  }
}
🧪 Testing & Training
Retraining the Brain
To retrain the Bandit model on new data:

Bash
python -m zone_1_training.trainer
Verify Model Integrity
To ensure your models haven't been tampered with:

Bash
python -m scripts.sign_models
📜 License
Distributed under the MIT License. See LICENSE for more information.

Developed for BSc Computer Science Final Year Project (2026).