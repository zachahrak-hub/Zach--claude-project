# Web Agent

An AI-powered web agent built with Flask and Claude. Supports browser automation, Excel questionnaire filling, and a document knowledge base.

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd web-agent
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure API key
```bash
cp .env.example .env
```
Then open `.env` and replace `your_api_key_here` with your [Anthropic API key](https://console.anthropic.com/).

### 5. Run the app
```bash
python app.py
```
Open http://localhost:5001 in your browser.

## Features
- **Browser Agent** — Give Claude a task and it controls a real browser
- **Excel Filler** — Upload a vendor questionnaire and company info to auto-fill it
- **Smart Fill** — Match two Excel files (questionnaire + company data) automatically
- **Knowledge Base** — Upload PDFs, Word docs, or Excel files and ask questions against them
