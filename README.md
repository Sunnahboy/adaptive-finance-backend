

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
## 🚀 Key Enterprise Features

* **⚡ Non-Blocking I/O (Async/Await):** Fully asynchronous architecture using FastAPI. LLM generation and Bandit math run in parallel (`asyncio.gather`), ensuring sub-second inference.
* **🧠 Hybrid LLM Multiplexer (True Round-Robin):** Engineered a zero-dependency, mathematically weighted load balancer using `itertools.cycle`. It seamlessly distributes 50/50 traffic between the native Google GenAI SDK (Gemini 2.5 Flash) and ultra-low-latency OpenRouter edge models (Mistral 24B, Llama 3B) to bypass rate limits and guarantee 100% uptime.
* **⚡ Deterministic Edge Logic:** Eliminated the "LLM-for-Math" anti-pattern. The system uses zero-latency, CPU-bound Python logic to evaluate Reinforcement Learning thresholds, slashing API overhead by 50% while reserving the LLMs strictly for dynamic copywriting.
* **🎯 Online Contextual Learning:** Implements `LinUCB` with Sherman-Morrison updates for $O(d^2)$ online learning efficiency. Features a FastAPI `BackgroundTasks` queue so the Bandit can update its matrix without lagging the mobile UI.
* **💾 High-Hit-Rate Caching Architecture:** Uses an asynchronous, WAL-mode SQLite cache (`aiosqlite`) to securely store prediction states. Cache keys are mathematically rounded (e.g., float precision grouping) to drastically increase cache hit rates, delivering 0.005s response times for recurring user contexts.
* **🛡️ Cryptographic Security:**
* **Model Integrity:** All AI models (`.pkl`) are digitally signed with HMAC-SHA256. The system performs an atomic file replacement (`os.replace`) and re-signs the model dynamically upon learning.
* **API Security:** Endpoints are protected via a custom `X-API-Token` header and strict `.env` driven CORS policies.


* **🛑 Resilient Fallback & Circuit Breaker:** Features a custom Async Circuit Breaker that monitors API latency in real-time. If a provider spikes above 4.0s, the circuit opens, instantly failing over to safe, hardcoded gamification states without trapping users in incorrect data flows.
* **🔄 Resilient MLOps Pipeline:** Engineered a custom micro-batching loop that asynchronously updates and synchronizes the Contextual Bandit model with Supabase Cloud Storage after a configurable threshold of user interactions.
* **☁️ Stateless Cloud Architecture:** Designed the AI memory to survive cloud server cold-starts and horizontal scaling by fetching the latest validated model state from the cloud upon server initialization.
* **🔐 Secure Artifact Management:** Implemented strict backend Row-Level Security (RLS) bypass mechanisms using protected `service_role` keys, ensuring public clients cannot tamper with the core AI model.

---

## 🏗️ System Architecture & Data Flow

The backend operates on a highly available, cloud-native architecture deployed on a dedicated AWS EC2 instance, designed for real-time reinforcement learning:

1. **API Gateway & Security:** Nginx Reverse Proxy with Let's Encrypt SSL (HTTPS). The AWS firewall strictly blocks direct port access (Port 8000), routing all traffic through Port 443.
2. **Infrastructure as Code (IaC):** The entire AWS environment is automated and deployed using **Ansible**.
3. **The LLM Routing Layer:** Incoming gamification requests hit the Round-Robin Multiplexer, instantly delegating the creative generation to the most available model in the pool (Gemini, Mistral, Llama, or Gemma) while maintaining the core contextual logic locally.
4. **Micro-Batched Learning:** To minimize network overhead, user feedback is cached locally via SQLite. The system batches interactions before triggering a high-priority cloud upload.
5. **Cloud Persistence:** Utilizes Supabase PostgreSQL for persistent MLOps analytics logging and Supabase Object Storage to host the globally synchronized model artifacts.

---

## 📂 File Structure

The project follows a strict **Micro-Zoned Architecture** to separate Data Science logic from Web Backend logic.


```text
adaptive_finance_ai/
├── Dockerfile                       # Containerization for cloud deployment
├── .env                             # Environment variables (API keys, CORS origins)
├── scripts/                         
│   └── sign_models.py               # HMAC-SHA256 cryptographic signature generator
├── shared_core/                     # Shared Domain Logic (DRY Principle)
│   ├── features.py                  # KBinsDiscretizer & Log1p pipelines
│   ├── llm_advisor.py               # Async Gemini integration & Circuit Breaker
│   ├── models.py                    # LinUCB Contextual Bandit implementation
│   └── schemas.py                   # Pydantic data validation API contracts
├── tests/                           # System and Integration Test Suite
│   └── test_llm.py                  # LLM routing and fallback tests
├── zone_1_training/                 # Offline ML Pipeline (Data Science Zone)
│   ├── data/                        # Tiered Data Engineering Pipeline
│   │   ├── raw/                     # Bronze: Original unprocessed data (online_retail_II)
│   │   ├── interim/                 # Silver: Partially cleaned transactions
│   │   └── processed/               # Gold: Final scaled ML-ready datasets
│   ├── notebooks/                   # Exploratory Data Analysis (EDA)
│   │   ├── 01_data_loading and cleaning.ipynb
│   │   └── 03_feature_engineering.ipynb
│   ├── src/                         
│   │   └── cmab_visualizations.py   # 9-panel matplotlib convergence dashboard
│   └── trainer.py                   # Main orchestrator (Teacher forcing, epsilon decay)
├── zone_2_artifacts/                # Model Vault (Secure Bridge)
│   ├── bandit_model.pkl             # Serialized LinUCB model weights
│   ├── bandit_model.pkl.sig         # HMAC-SHA256 integrity signature
│   └── training_config.json         # Hyperparameters and action maps
└── zone_3_inference/                # Production Backend (FastAPI Service)
    └── app/                         
        ├── main.py                  # API gateway, CORS, and background tasks
        ├── services/                
        │   ├── leaderboard_service.py # MLOps analytics logging
        │   └── prediction_service.py  # Bandit inference + LLM routing caching
        └── templates/               
            └── dashboard.html       # Chart.js admin dashboard

```

---

## 🛠️ Getting Started

### Prerequisites

* Python 3.10+
* Google AI Studio API Key (for Gemini)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/adaptive-finance-backend.git
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
API_ACCESS_TOKEN=my_thesis_access_token_2026
MODEL_SIGNING_KEY=my_super_secret_thesis_key_2026

# --- CORS Rules ---
ALLOWED_ORIGINS=http://localhost:8000,http://10.0.2.2:8000,https://adaptivefinance.duckdns.org

```

### 3. Run Locally

```bash
python zone_3_inference/app/main.py

```

Visit `http://127.0.0.1:8000/docs` to access the local Swagger UI. For the live production environment, visit `https://adaptivefinance.duckdns.org/docs`.

---

## 📡 API Usage (The Stateless Loop)

The system operates on a two-step reinforcement learning loop.

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

**Endpoint:** `POST /predict/v1/feedback`
**Headers:** `Content-Type: application/json` | `X-API-Token: <YOUR_ACCESS_TOKEN>`

**Request:**

```json
{
  "prediction_id": "e0c9ed02-d764-40e9-b85f-43bb354e104b",
  "reward": 1.0 
}

```

*(Returns a `202 Accepted` status immediately, while the Bandit matrix updates and atomically re-signs its `.pkl` file in a background worker thread).*

---

## 📊 Live Monitoring Dashboard

The system includes a zero-configuration MLOps dashboard to monitor the AI's real-time performance without heavy external dependencies.

**To access the dashboard:**
Navigate to: `https://adaptivefinance.duckdns.org/admin/dashboard`

**Tracked Metrics:**

* **Action Distribution:** A pie chart showing the percentage of traffic allocated to each gamification strategy.
* **Success Rate (CTR):** A bar chart tracking the real-world reward conversion rate of each arm, proving the Bandit is identifying the optimal strategy.

---

## ⚙️ Developer Command Cheat Sheet

Below is the complete log of terminal commands used to provision, secure, and deploy the AWS EC2 infrastructure from a local Windows Subsystem for Linux (WSL) environment.

### 1. Local WSL & SSH Key Preparation

```bash
# Copy the downloaded AWS key into the Linux home directory
cp /mnt/c/Users/User/Downloads/fyp-key.pem ~

# Lock the file permissions (Read-only for the owner)
chmod 400 ~/fyp-key.pem

# Establish a secure SSH tunnel into the EC2 instance
ssh -i "~/fyp-key.pem" ubuntu@3.26.148.159

```

### 2. Infrastructure as Code (Ansible)

```bash
# Install Ansible on the local WSL machine
sudo apt update && sudo apt install ansible -y

# Test the connection to the AWS server
ansible -i inventory.ini all -m ping

# Execute the master deployment playbook
ansible-playbook -i inventory.ini setup.yml

```

### 3. Server Monitoring & Debugging

```bash
# Follow the live, real-time logs of the AI backend daemon
sudo journalctl -u adaptive-finance -f

```

### 4. Reverse Proxy & SSL Configuration (Nginx)

```bash
# Install Nginx and Certbot on AWS
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx -y

# Create symlink and remove default placeholder
sudo ln -s /etc/nginx/sites-available/adaptivefinance /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx syntax and restart
sudo nginx -t
sudo systemctl restart nginx

# Generate SSL certificate and enforce HTTPS
sudo certbot --nginx -d adaptivefinance.duckdns.org

```

---

📜 **License:** Distributed under the MIT License.

🎓 **Developed for:** BSc Computer Science Final Year Project (2026) - Asia Pacific University of Technology and Innovation.

