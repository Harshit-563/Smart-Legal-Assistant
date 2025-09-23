🧠 Smart Legal Assistant

Smart Legal Assistant is an NLP-powered application that helps you analyze legal contracts, extract key clauses, generate concise summaries, and flag risky terms — making contract review faster and smarter.

🚀 Features

Clause Extraction: Automatically identify important clauses in contracts.

Risk Detection: Highlight risky terms or unusual conditions.

Summarization: Generate easy-to-read contract summaries.

FastAPI Powered: Lightweight, fast, and scalable backend.

Containerized: Run anywhere with Docker.

🛠️ Tech Stack

Python: 3.10 (slim)

FastAPI: Web framework

NLP Libraries: SpaCy, Transformers, or others (from requirements.txt)

Containerization: Docker & Docker Compose

⚡ Quick Start
1️⃣ Local Setup

Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate


Install dependencies:

pip install -r docs/requirements.txt


Run the app in development mode:

uvicorn src.smart_legal_assistant:app --reload --port 8000


Quick test:

python app.py

2️⃣ Run with Docker

The project includes a Dockerfile and docker-compose.yml for easy containerized setup.

Build and start the app:

docker compose up --build


The FastAPI app will be available at: http://localhost:8000

Service name: python-app

Restart policy: Automatically restart unless stopped

Optional: Add a .env file and uncomment the env_file line in docker-compose.yml if you need custom environment variables.

📂 Project Structure
.
├── src/
│   └── smart_legal_assistant.py    # Main FastAPI app
├── docs/
│   └── requirements.txt            # Python dependencies
├── app.py                          # Quick test script
├── Dockerfile
└── docker-compose.yml

💡 Usage

Start the app (locally or via Docker)

Send legal documents via API endpoints

Receive:

Extracted clauses

Risk warnings

Summaries

🌟 Contributing

Contributions are welcome!

Fork the repository

Create a feature branch

Submit a pull request

📄 License

This project is licensed under MIT License.
