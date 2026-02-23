import io
import json
import os
import tempfile
import re

import anthropic
import openpyxl
from dotenv import load_dotenv
from functools import wraps
from flask import Flask, jsonify, render_template, request, send_file, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from playwright.sync_api import sync_playwright

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

limiter = Limiter(get_remote_address, app=app, default_limits=[])

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Browser state ──────────────────────────────────────────────────────────────
# Playwright must run on the same thread - use threading.local
import threading
_local = threading.local()


def get_page():
    if not getattr(_local, "page", None):
        _local.pw = sync_playwright().start()
        _local.browser = _local.pw.chromium.launch(headless=True)
        _local.page = _local.browser.new_page()
    return _local.page


def close_browser():
    if getattr(_local, "browser", None):
        _local.browser.close()
    if getattr(_local, "pw", None):
        _local.pw.stop()
    _local.pw = None
    _local.browser = None
    _local.page = None


# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "navigate",
        "description": "Navigate to a given URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "click",
        "description": "Click an element by CSS selector",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector of the element"}
            },
            "required": ["selector"],
        },
    },
    {
        "name": "fill",
        "description": "Fill a text field by CSS selector",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector of the field"},
                "value": {"type": "string", "description": "Value to fill in"},
            },
            "required": ["selector", "value"],
        },
    },
    {
        "name": "get_page_content",
        "description": "Read the current page content (text only, up to 3000 characters)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "screenshot",
        "description": "Take a screenshot of the current screen and describe it",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_url",
        "description": "Quickly fetch the raw text content of a URL without a browser (fast, use for plain text files like llms.txt or simple HTML pages)",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape)",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name"}
            },
            "required": ["key"],
        },
    },
]

# ── Excel tools ────────────────────────────────────────────────────────────────
EXCEL_TOOLS = [
    {
        "name": "fill_excel_cell",
        "description": "Fill a specific Excel cell by sheet name, row and column",
        "input_schema": {
            "type": "object",
            "properties": {
                "sheet": {"type": "string", "description": "Sheet name"},
                "row": {"type": "integer", "description": "Row number (1-based)"},
                "col": {"type": "integer", "description": "Column number (1-based)"},
                "value": {"type": "string", "description": "Value to fill"},
            },
            "required": ["sheet", "row", "col", "value"],
        },
    },
    {
        "name": "get_excel_structure",
        "description": "Get the Excel structure - list of sheets and questions",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ── Tool executor (browser) ────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    page = get_page()
    try:
        if tool_name == "navigate":
            page.goto(tool_input["url"], timeout=15000)
            return f"✅ Navigated to: {tool_input['url']}"
        elif tool_name == "click":
            page.click(tool_input["selector"], timeout=8000)
            return f"✅ Clicked: {tool_input['selector']}"
        elif tool_name == "fill":
            page.fill(tool_input["selector"], tool_input["value"], timeout=8000)
            return f"✅ Filled '{tool_input['selector']}' with: {tool_input['value']}"
        elif tool_name == "fetch_url":
            import requests as req
            try:
                r = req.get(tool_input["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                return r.text[:5000]
            except Exception as e:
                return f"❌ Fetch error: {str(e)}"
        elif tool_name == "get_page_content":
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return page.inner_text("body")[:5000]
        elif tool_name == "screenshot":
            page.screenshot(path="/tmp/agent_screenshot.png")
            return "✅ Screenshot saved"
        elif tool_name == "press_key":
            page.keyboard.press(tool_input["key"])
            return f"✅ Pressed key: {tool_input['key']}"
        else:
            return f"❌ Unknown tool: {tool_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Excel helpers ──────────────────────────────────────────────────────────────
def read_excel_questions(wb: openpyxl.Workbook) -> str:
    """Returns a description of all questions in the file with their exact positions."""
    output = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        output.append(f"\n=== Sheet: {sheet_name} ===")
        for row in ws.iter_rows():
            for cell in row:
                if cell.value and isinstance(cell.value, str) and len(cell.value) > 3:
                    output.append(
                        f"  Row {cell.row}, Col {cell.column} -> Question: {cell.value[:120]}"
                    )
    return "\n".join(output)


def execute_excel_tool(tool_name: str, tool_input: dict, wb: openpyxl.Workbook) -> str:
    if tool_name == "get_excel_structure":
        return read_excel_questions(wb)

    elif tool_name == "fill_excel_cell":
        sheet = tool_input["sheet"]
        row = tool_input["row"]
        col = tool_input["col"]
        value = tool_input["value"]
        if sheet not in wb.sheetnames:
            return f"❌ Sheet '{sheet}' not found"
        wb[sheet].cell(row=row, column=col, value=value)
        return f"✅ Filled [{sheet}] row {row}, col {col}: {value[:60]}"

    return f"❌ Unknown tool: {tool_name}"


# ── Coralogix fast-path helpers ────────────────────────────────────────────────
def _cx_find_urls(question: str, llms_text: str, max_urls: int = 2) -> list:
    """Score llms.txt URLs by keyword overlap with the question."""
    import re as _re
    keywords = set(_re.findall(r'\b\w{4,}\b', question.lower()))
    scored = []
    for line in llms_text.splitlines():
        m = _re.search(r'https?://[^\s\)]+', line)
        if not m:
            continue
        url = m.group(0).rstrip('.,)')
        score = sum(1 for kw in keywords if kw in line.lower())
        if score > 0:
            scored.append((score, url))
    scored.sort(reverse=True)
    seen, result = set(), []
    for _, url in scored:
        if url not in seen:
            seen.add(url)
            result.append(url)
        if len(result) >= max_urls:
            break
    return result


def _cx_fetch(url: str) -> str:
    import requests as req
    try:
        r = req.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        return r.text[:5000]
    except Exception:
        return ""


def coralogix_direct_answer(question: str) -> dict:
    """Answer a Coralogix question by scraping the actual docs pages."""
    import concurrent.futures, requests as req

    HEADERS = {"User-Agent": "Mozilla/5.0"}

    # 1. Fetch the full docs index (118k chars — fetch it all)
    try:
        r = req.get("https://coralogix.com/docs/llms.txt", timeout=10, headers=HEADERS)
        llms_text = r.text  # full file, no truncation
    except Exception:
        return None

    # 2. Smart Haiku call: understand the question semantics, pick best URLs
    pick = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=300,
        messages=[{"role": "user", "content":
            f"You help find the right Coralogix documentation pages.\n\n"
            f"Question: \"{question}\"\n\n"
            f"Important hints:\n"
            f"- 'AWS locations', 'regions', 'domains' → look for 'Coralogix Domain' or 'account-settings'\n"
            f"- 'pricing', 'plans' → look for billing/payment pages\n"
            f"- 'train', 'AI', 'data privacy' → look for privacy/security pages\n"
            f"- For AWS-specific integrations (CloudWatch, S3, etc.) → look under /integrations/aws/\n\n"
            f"From the index below, return the 2-3 best page URLs to answer the question.\n"
            f"Reply with ONLY the full URLs (starting with https://), one per line.\n\n"
            f"INDEX:\n{llms_text[:30000]}"}],
    )
    url_text = next((b.text for b in pick.content if hasattr(b, "text")), "")
    urls = [l.strip() for l in url_text.splitlines()
            if l.strip().startswith("https://coralogix.com/docs")][:3]
    if not urls:
        return None

    # 3. Fetch all candidate pages in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_cx_fetch, urls))

    pages = {url: content for url, content in zip(urls, results) if content.strip()}
    if not pages:
        return None

    # 4. Sonnet: write the answer from the real page content
    context = "\n\n".join(f"=== {url} ===\n{content}" for url, content in pages.items())
    sources = "\n".join(f"- {url}" for url in pages)

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content":
            f"Answer this question using ONLY the Coralogix documentation below.\n\n"
            f"Question: {question}\n\n"
            f"Documentation:\n{context}\n\n"
            f"Write a clear, natural answer (2-4 paragraphs) based strictly on the docs above.\n"
            f"End with:\n\n📎 Sources:\n{sources}"}],
    )
    answer = next((b.text for b in response.content if hasattr(b, "text")), "")
    steps = [{"tool": "fetch_url", "input": {"url": u}, "result": "fetched"} for u in pages]
    return {"result": answer, "steps": steps}


# ── Agent route (browser) ──────────────────────────────────────────────────────
@app.route("/agent", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
def run_agent():
    data = request.get_json()
    task = data.get("task", "")
    if not task:
        return jsonify({"error": "No task received"}), 400

    # Fast path: Coralogix questions answered without agentic loop
    cx_keywords = ["coralogix"]
    if any(kw in task.lower() for kw in cx_keywords):
        result = coralogix_direct_answer(task)
        if result:
            return jsonify(result)

    # Standard agentic loop for everything else
    messages = [{"role": "user", "content": task}]
    steps = []

    system_prompt = """You are an AI assistant that performs browser tasks.

When the user asks about weather, weather forecast, temperature, rain, wind,
or any weather-related topic in Israel - always navigate first to the Israeli
Meteorological Service website: https://ims.gov.il and read the data there."""

    for _ in range(20):
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            result = execute_tool(tool_block.name, tool_block.input)
            steps.append({"tool": tool_block.name, "input": tool_block.input, "result": result})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": result}],
            })
        else:
            final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
            return jsonify({"result": final_text, "steps": steps})

    return jsonify({"result": "Reached iteration limit", "steps": steps})


# ── Excel agent route ──────────────────────────────────────────────────────────
@app.route("/fill-excel", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
def fill_excel():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    company_info = request.form.get("info", "")

    if not company_info:
        return jsonify({"error": "No company information provided"}), 400

    wb = openpyxl.load_workbook(io.BytesIO(file.read()))

    system_prompt = """You are a professional vendor questionnaire specialist.
You will receive an Excel file structure with questions and company information.
Fill the Response column (usually column 2) for each question.
Use fill_excel_cell to fill each answer.
Start with get_excel_structure to see the questions.
Write professional, concise, clear answers in English.
If no relevant information is available - write 'N/A' or 'To be provided'."""

    user_message = f"""Fill the following vendor questionnaire based on the company information.

Company information:
{company_info}

Start with get_excel_structure to see the questions, then fill each one."""

    messages = [{"role": "user", "content": user_message}]
    steps = []

    for _ in range(60):  # Long questionnaires require more iterations
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=system_prompt,
            tools=EXCEL_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            result = execute_excel_tool(tool_block.name, tool_block.input, wb)
            steps.append({"tool": tool_block.name, "input": tool_block.input, "result": result})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": result}],
            })
        else:
            output = io.BytesIO()
            wb.save(output)
            output.seek(0)

            original_name = file.filename.rsplit(".", 1)[0]
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name=f"{original_name}_filled.xlsx",
            )

    return jsonify({"error": "Reached iteration limit"}), 500


@app.route("/close", methods=["POST"])
@login_required
def close():
    close_browser()
    return jsonify({"status": "Browser closed"})


@app.route("/download-template")
@login_required
def download_template():
    """Download the company_data_template.xlsx template"""
    path = os.path.join(os.path.dirname(__file__), "static", "company_data_template.xlsx")
    return send_file(path, as_attachment=True, download_name="company_data_template.xlsx")


# ── Smart Fill: questionnaire + company data ───────────────────────────────────
def read_excel_as_text(wb: openpyxl.Workbook, max_chars: int = 6000) -> str:
    """Reads a workbook and returns structured text - limited to max_chars"""
    lines = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"\n=== Sheet: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            row_vals = [str(c).strip() if c is not None else "" for c in row]
            if any(v for v in row_vals):
                lines.append(" | ".join(row_vals[:6]))  # max 6 columns
    text = "\n".join(lines)
    return text[:max_chars]


def get_questionnaire_questions(wb: openpyxl.Workbook) -> list:
    """Returns a list of (sheet, row, col, question) for all questions"""
    questions = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for row in ws.iter_rows():
            for cell in row:
                val = cell.value
                if val and isinstance(val, str) and len(val) > 10:
                    if any(c.isalpha() for c in val) and "?" not in val[:3]:
                        resp_col = cell.column + 1
                        if resp_col <= ws.max_column:
                            resp_cell = ws.cell(row=cell.row, column=resp_col)
                            if resp_cell.value is None or resp_cell.value == "":
                                questions.append({
                                    "sheet": sheet_name,
                                    "row": cell.row,
                                    "col": resp_col,
                                    "question": val[:200]
                                })
    return questions


@app.route("/smart-fill", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
def smart_fill():
    """Auto-fill an Excel questionnaire based on a company data file"""
    import time

    if "questionnaire" not in request.files or "company_data" not in request.files:
        return jsonify({"error": "Two files required: questionnaire + company_data"}), 400

    q_file = request.files["questionnaire"]
    c_file = request.files["company_data"]

    q_wb = openpyxl.load_workbook(io.BytesIO(q_file.read()))
    c_wb = openpyxl.load_workbook(io.BytesIO(c_file.read()))

    # Read company profile (truncated)
    company_text = read_excel_as_text(c_wb, max_chars=5000)

    # Read questionnaire structure (truncated)
    questionnaire_text = read_excel_as_text(q_wb, max_chars=5000)

    system_prompt = """You are an expert vendor security questionnaire specialist.
You have a company profile and a vendor questionnaire to fill.
- Use fill_excel_cell to write answers into empty Response columns
- For Yes/No questions answer exactly "Yes" or "No"
- Keep answers concise and professional
- If info not available write "To be provided"
- Do NOT modify question columns"""

    user_message = f"""Fill this vendor questionnaire using the company profile.

COMPANY PROFILE (summary):
{company_text}

QUESTIONNAIRE LAYOUT:
{questionnaire_text}

Use get_excel_structure first, then fill_excel_cell for each empty response cell."""

    messages = [{"role": "user", "content": user_message}]

    for _ in range(80):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                system=system_prompt,
                tools=EXCEL_TOOLS,
                messages=messages,
            )
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(60)  # Wait a minute and retry
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2048,
                    system=system_prompt,
                    tools=EXCEL_TOOLS,
                    messages=messages,
                )
            else:
                return jsonify({"error": str(e)}), 500

        if response.stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            result = execute_excel_tool(tool_block.name, tool_block.input, q_wb)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": result}],
            })
        else:
            output = io.BytesIO()
            q_wb.save(output)
            output.seek(0)
            original_name = q_file.filename.rsplit(".", 1)[0]
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name=f"{original_name}_filled.xlsx",
            )

    return jsonify({"error": "Reached iteration limit"}), 500


# ── Knowledge Base ─────────────────────────────────────────────────────────────
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)


def extract_text_from_file(filepath: str, filename: str) -> str:
    """Extract plain text from PDF, Excel, or Word file."""
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in ("xlsx", "xls"):
            wb = openpyxl.load_workbook(filepath)
            lines = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows(values_only=True):
                    vals = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if vals:
                        lines.append(" | ".join(vals))
            return "\n".join(lines)
        elif ext in ("docx",):
            from docx import Document
            doc = Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"
    return ""


def load_knowledge_base() -> str:
    """Load all documents from the knowledge base folder into a single text."""
    chunks = []
    for fname in os.listdir(KB_DIR):
        fpath = os.path.join(KB_DIR, fname)
        if os.path.isfile(fpath):
            text = extract_text_from_file(fpath, fname)
            if text.strip():
                chunks.append(f"\n\n===== Document: {fname} =====\n{text[:8000]}")
    return "\n".join(chunks)


@app.route("/upload-kb", methods=["POST"])
@login_required
def upload_kb():
    """Upload a document to the knowledge base."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    save_path = os.path.join(KB_DIR, f.filename)
    f.save(save_path)
    return jsonify({"status": f"✅ '{f.filename}' added to knowledge base"})


@app.route("/list-kb", methods=["GET"])
@login_required
def list_kb():
    """List all documents in the knowledge base."""
    files = [f for f in os.listdir(KB_DIR) if os.path.isfile(os.path.join(KB_DIR, f))]
    return jsonify({"files": files})


@app.route("/delete-kb", methods=["POST"])
@login_required
def delete_kb():
    """Delete a document from the knowledge base."""
    data = request.get_json()
    fname = data.get("filename", "")
    fpath = os.path.join(KB_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        return jsonify({"status": f"✅ '{fname}' deleted"})
    return jsonify({"error": "File not found"}), 404


@app.route("/ask-kb", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
def ask_kb():
    """Answer a question using the knowledge base documents."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    kb_text = load_knowledge_base()
    if not kb_text.strip():
        return jsonify({"error": "Knowledge base is empty. Please upload documents first."}), 400

    system_prompt = """You are a professional security and compliance assistant for Coralogix.
You answer questions strictly based on the provided documents from Coralogix's knowledge base.
- Give clear, accurate, and professional answers
- Quote or reference the specific document/section when possible
- If the answer is not found in the documents, say so clearly
- Keep answers concise but complete
- Format your answer in plain text, no markdown"""

    user_message = f"""Answer the following question using only the documents provided below.

QUESTION:
{question}

KNOWLEDGE BASE DOCUMENTS:
{kb_text[:20000]}"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = next((b.text for b in response.content if hasattr(b, "text")), "No answer found.")
    return jsonify({"answer": answer})


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if (username == os.getenv("APP_USERNAME", "admin") and
                password == os.getenv("APP_PASSWORD", "changeme")):
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/health")
def health():
    return "ok", 200


@app.route("/")
@login_required
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
