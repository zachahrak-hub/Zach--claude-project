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
            # Return JSON error for API routes, redirect for page routes
            if request.is_json or request.method == "POST":
                return jsonify({"error": "Session expired. Please refresh and log in again."}), 401
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
import time as _time

_llms_cache = {"text": None, "ts": 0}
_LLMS_TTL = 3600  # cache llms.txt for 1 hour


def _get_llms_txt() -> str:
    """Return cached llms.txt, re-fetching only if older than 1 hour."""
    import requests as req
    now = _time.time()
    if _llms_cache["text"] and (now - _llms_cache["ts"]) < _LLMS_TTL:
        return _llms_cache["text"]
    try:
        r = req.get("https://coralogix.com/docs/llms.txt", timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"})
        _llms_cache["text"] = r.text
        _llms_cache["ts"] = now
        return r.text
    except Exception:
        return _llms_cache["text"] or ""


def _cx_fetch(url: str) -> str:
    import requests as req
    try:
        r = req.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        return r.text[:3000]
    except Exception:
        return ""


def coralogix_direct_answer(question: str) -> dict:
    """Answer a Coralogix question. Never returns None — always gives an answer."""
    import concurrent.futures, requests as req

    HEADERS = {"User-Agent": "Mozilla/5.0"}
    pages = {}

    # 1. Try to find and fetch relevant docs pages
    try:
        llms_text = _get_llms_txt()
        if llms_text:
            pick = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                messages=[{"role": "user", "content":
                    f"Find the best Coralogix docs URLs for this question.\n"
                    f"Question: \"{question}\"\n\n"
                    f"Hints: SOC2/ISO/certifications → security pages, "
                    f"AWS → integrations/aws, privacy/AI → privacy pages.\n\n"
                    f"Reply with ONLY 2-3 URLs starting with https://, one per line.\n\n"
                    f"INDEX:\n{llms_text[:30000]}"}],
            )
            url_text = next((b.text for b in pick.content if hasattr(b, "text")), "")
            urls = [l.strip() for l in url_text.splitlines()
                    if l.strip().startswith("https://coralogix.com")][:3]

            if urls:
                def fetch(url):
                    try:
                        r = req.get(url, timeout=8, headers=HEADERS)
                        return r.text[:5000]
                    except Exception:
                        return ""
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                    results = list(ex.map(fetch, urls))
                pages = {url: c for url, c in zip(urls, results) if c.strip()}
    except Exception:
        pass  # Fall through to knowledge-based answer

    # 2. Build answer — from docs if available, from training knowledge otherwise
    if pages:
        context = "\n\n".join(f"=== {url} ===\n{c}" for url, c in pages.items())
        sources = "\n".join(f"- {url}" for url in pages)
        user_msg = (f"Answer this question using the Coralogix docs below.\n"
                    f"Question: {question}\n\n"
                    f"Docs:\n{context}\n\n"
                    f"Write a clear answer (2-4 paragraphs).\n"
                    f"End with:\n\n📎 Sources:\n{sources}")
    else:
        user_msg = (f"Answer this question about Coralogix based on your knowledge.\n"
                    f"Question: {question}\n\n"
                    f"Write a clear, accurate answer (2-4 paragraphs).\n"
                    f"End with:\n\n📎 Sources:\n- https://coralogix.com/docs/")

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer = next((b.text for b in response.content if hasattr(b, "text")), "")
    steps = [{"tool": "fetch_url", "input": {"url": u}, "result": "fetched"} for u in pages]
    return {"result": answer, "steps": steps}


# ── Agent route (browser) ──────────────────────────────────────────────────────
@app.route("/agent", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def run_agent():
    data = request.get_json()
    task = data.get("task", "")
    if not task:
        return jsonify({"error": "No task received"}), 400

    # Fast path: Coralogix questions answered without agentic loop
    cx_keywords = ["coralogix"]
    if any(kw in task.lower() for kw in cx_keywords):
        try:
            result = coralogix_direct_answer(task)
            if result:
                return jsonify(result)
            return jsonify({"result": "Could not find a relevant answer in the Coralogix documentation. Please try rephrasing your question.", "steps": []})
        except Exception as e:
            return jsonify({"result": f"Error while searching Coralogix docs: {str(e)}", "steps": []})

    # Standard agentic loop for everything else
    messages = [{"role": "user", "content": task}]
    steps = []

    system_prompt = """You are an AI assistant that performs browser tasks.

When the user asks about weather, weather forecast, temperature, rain, wind,
or any weather-related topic in Israel - always navigate first to the Israeli
Meteorological Service website: https://ims.gov.il and read the data there."""

    try:
        for _ in range(20):
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_results = []
                for tool_block in tool_blocks:
                    result = execute_tool(tool_block.name, tool_block.input)
                    steps.append({"tool": tool_block.name, "input": tool_block.input, "result": result})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result
                    })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return jsonify({"result": final_text, "steps": steps})
    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}", "steps": steps})

    return jsonify({"result": "Reached iteration limit", "steps": steps})


# ── Vendor Vetting route ───────────────────────────────────────────────────────
@app.route("/vet-vendor", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def vet_vendor():
    """Auto-research a vendor company and generate a security vetting report."""
    import concurrent.futures
    try:
        return _vet_vendor_impl()
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-500:]}), 500


def _vet_vendor_impl():
    import concurrent.futures

    company_name = ""

    # If file uploaded — extract company name from it
    if "file" in request.files:
        f = request.files["file"]
        suffix = "." + f.filename.rsplit(".", 1)[-1] if "." in f.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        try:
            raw_text = extract_text_from_file(tmp_path, f.filename)[:3000]
        finally:
            os.unlink(tmp_path)
        extract_resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content":
                f"Extract only the vendor/company name from this document. "
                f"Reply with just the company name, nothing else.\n\n{raw_text}"}]
        )
        company_name = next((b.text.strip() for b in extract_resp.content if hasattr(b, "text")), "")
    else:
        data = request.get_json(silent=True) or {}
        company_name = data.get("company", "").strip()

    if not company_name:
        return jsonify({"error": "Company name required"}), 400

    # Step 1: Use Haiku to generate likely security/trust URLs for the company
    url_resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=400,
        messages=[{"role": "user", "content":
            f"For the company '{company_name}', generate 8-10 likely URLs.\n"
            f"PRIORITY — Trust Center (these are most important):\n"
            f"  trust.<domain>.com, security.<domain>.com, <domain>.com/trust, <domain>.com/security\n"
            f"Also include: homepage, /privacy, /legal, /compliance, /certifications\n"
            f"Reply with ONLY full URLs starting with https://, one per line, no extra text."}]
    )
    url_text = next((b.text for b in url_resp.content if hasattr(b, "text")), "")
    urls = [l.strip() for l in url_text.splitlines() if l.strip().startswith("https://")][:10]

    if not urls:
        return jsonify({"error": f"Could not determine URLs for '{company_name}'"}), 500

    # Step 2: Fetch all URLs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_cx_fetch, urls))

    pages = {url: content for url, content in zip(urls, results) if content.strip()}
    context = "\n\n".join(f"=== {url} ===\n{content}" for url, content in pages.items()) if pages else "No pages could be fetched."

    # Step 3: Generate structured vetting report + extract trust center & docs
    report_resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{"role": "user", "content":
            f"You are a vendor security analyst. Generate a structured vetting report for '{company_name}' "
            f"based on the web pages below.\n\n"
            f"Include these sections:\n"
            f"## Company Overview\n"
            f"## Security Certifications (SOC 2, ISO 27001, PCI-DSS, etc.)\n"
            f"## Data Privacy & GDPR\n"
            f"## Data Security Practices\n"
            f"## Known Issues or Risks\n\n"
            f"## 🎯 Key Vetting Criteria\n"
            f"For each criterion below, answer with ✅ Pass / ❌ Fail / ⚠️ Unclear — followed by a one-line explanation:\n"
            f"1. **Data Training** — Does the vendor train AI/ML models on customer data?\n"
            f"2. **Data Ownership** — Is it explicitly stated that customer data always belongs to the customer?\n"
            f"3. **Data Retention** — Is retention minimized and strictly limited to the stated purpose only?\n"
            f"4. **Legal Risk** — Is the vendor classifiable as low risk by legal/regulatory definition?\n\n"
            f"## Overall Verdict — state Low / Medium / High risk and a one-line summary\n\n"
            f"Be concise. If info not found on the pages, say 'Not found on public pages'.\n\n"
            f"WEB PAGES:\n{context}"}]
    )
    report = next((b.text for b in report_resp.content if hasattr(b, "text")), "No report generated.")

    # Step 4: Extract trust center URL and document links from fetched pages
    links_resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=400,
        messages=[{"role": "user", "content":
            f"From the web pages below, extract:\n"
            f"1. The Trust Center URL (trust center, security portal, compliance hub)\n"
            f"2. Up to 5 direct document links (SOC 2 report, ISO 27001 certificate, pen-test, DPA, etc.)\n\n"
            f"Reply in this exact JSON format (no extra text):\n"
            f'{{"trust_center": "https://..." or null, "documents": [{{"name": "SOC 2 Report", "url": "https://..."}}]}}\n\n'
            f"WEB PAGES:\n{context[:8000]}"}]
    )
    links_text = next((b.text for b in links_resp.content if hasattr(b, "text")), "{}")
    try:
        import json as json_lib
        links_data = json_lib.loads(links_text.strip())
    except Exception:
        links_data = {"trust_center": None, "documents": []}

    return jsonify({
        "company": company_name,
        "report": report,
        "sources": list(pages.keys()),
        "trust_center": links_data.get("trust_center"),
        "documents": links_data.get("documents", []),
    })


# ── Excel agent route ──────────────────────────────────────────────────────────
@app.route("/fill-excel", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
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
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for tool_block in tool_blocks:
                result = execute_excel_tool(tool_block.name, tool_block.input, wb)
                steps.append({"tool": tool_block.name, "input": tool_block.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result
                })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
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
    """Returns a list of {sheet, row, col, question} for all unanswered questions.

    Strategy:
    1. Scan the first 6 rows for a header row containing both a 'question' column
       and a 'response' column.
    2. Use those column indices to find rows with a question and an empty response.
    3. If no clear header pair is found, fall back to a conservative heuristic that
       only picks up long descriptive text (likely real questions) with an empty
       adjacent cell.
    """
    QUESTION_KEYWORDS = {"question", "request", "question/request", "item",
                         "requirement", "description", "questionnaire"}
    RESPONSE_KEYWORDS = {"response", "answer", "vendor response", "your response",
                         "vendor answer", "reply", "please respond"}
    PLACEHOLDER_VALUES = {"", "to be provided", "n/a", "-", "tbd", "pending",
                          "please provide", "[to be filled]", "none"}

    questions = []
    seen = set()

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if not ws.max_row:
            continue

        # ── Step 1: find header row ────────────────────────────────────────────
        # Require BOTH a question column AND a response column in the SAME row.
        # Among question-keyword matches, prefer the longest header (more specific).
        q_col = None   # column index of the 'Question' header
        r_col = None   # column index of the 'Response' header
        header_row_idx = None

        for row_idx in range(1, min(8, ws.max_row + 1)):
            row_q_col = None
            row_q_len = 0   # track header length to prefer specific matches
            row_r_col = None
            for cell in ws[row_idx]:
                if not cell.value or not isinstance(cell.value, str):
                    continue
                val_lower = cell.value.strip().lower()
                # Question column: prefer the longest matching header in the row
                if any(kw in val_lower for kw in QUESTION_KEYWORDS):
                    if row_q_col is None or len(cell.value) > row_q_len:
                        row_q_col = cell.column
                        row_q_len = len(cell.value)
                # Response column: first match wins
                if row_r_col is None and any(kw in val_lower for kw in RESPONSE_KEYWORDS):
                    row_r_col = cell.column
            # Only accept this row as the header if BOTH columns were found and differ
            if row_q_col and row_r_col and row_q_col != row_r_col:
                q_col = row_q_col
                r_col = row_r_col
                header_row_idx = row_idx
                break

        # ── Step 2a: header-based scan ────────────────────────────────────────
        if q_col and r_col and q_col != r_col:
            start_row = (header_row_idx or 1) + 1
            for row_idx in range(start_row, ws.max_row + 1):
                q_val = ws.cell(row=row_idx, column=q_col).value
                r_val = ws.cell(row=row_idx, column=r_col).value

                if not (q_val and isinstance(q_val, str) and len(q_val.strip()) > 5):
                    continue
                if not any(c.isalpha() for c in q_val):
                    continue

                r_str = str(r_val).strip().lower() if r_val is not None else ""
                if r_str in PLACEHOLDER_VALUES:
                    key = (sheet_name, row_idx, r_col)
                    if key not in seen:
                        seen.add(key)
                        questions.append({
                            "sheet": sheet_name,
                            "row": row_idx,
                            "col": r_col,
                            "question": q_val.strip()[:200],
                        })

        # ── Step 2b: conservative fallback (no clear headers) ─────────────────
        else:
            SKIP_EXACT = {"response", "question", "question/request", "required clarification",
                          "additional information/comments", "coralogix review & comments",
                          "general", "security", "gdpr compliance"}
            for row in ws.iter_rows():
                for cell in row:
                    val = cell.value
                    # Only pick up clearly question-like text: >20 chars or contains "?"
                    if not (val and isinstance(val, str) and (len(val) > 20 or "?" in val)):
                        continue
                    if val.strip().lower() in SKIP_EXACT:
                        continue
                    if not any(c.isalpha() for c in val):
                        continue
                    resp_col = cell.column + 1
                    if resp_col > ws.max_column + 1:
                        continue
                    resp_cell = ws.cell(row=cell.row, column=resp_col)
                    r_str = str(resp_cell.value).strip().lower() if resp_cell.value is not None else ""
                    if r_str in PLACEHOLDER_VALUES:
                        key = (sheet_name, cell.row, resp_col)
                        if key not in seen:
                            seen.add(key)
                            questions.append({
                                "sheet": sheet_name,
                                "row": cell.row,
                                "col": resp_col,
                                "question": val[:200],
                            })

    return questions


@app.route("/smart-fill", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def smart_fill():
    """Auto-fill an Excel questionnaire using the Knowledge Base as company data"""
    import time

    if "questionnaire" not in request.files:
        return jsonify({"error": "Questionnaire file required"}), 400

    q_file = request.files["questionnaire"]
    q_wb = openpyxl.load_workbook(io.BytesIO(q_file.read()))

    # Load company data from the Knowledge Base
    company_text = load_knowledge_base()
    if not company_text.strip():
        return jsonify({"error": "Knowledge Base is empty. Please upload company documents in the Knowledge Base tab first."}), 400

    company_text = company_text[:8000]

    # Read questionnaire structure with exact row/col positions
    questions = get_questionnaire_questions(q_wb)
    if not questions:
        return jsonify({"error": "No questions found in the Excel file. Make sure the file has questions with empty response columns."}), 400

    # Build explicit fill list for the AI
    fill_instructions = "\n".join(
        f'- Sheet="{q["sheet"]}", Row={q["row"]}, Col={q["col"]}, Question: {q["question"]}'
        for q in questions
    )

    system_prompt = """You are an expert vendor security questionnaire specialist.
For EACH question listed, call fill_excel_cell with the exact sheet, row, col, and a concise answer.
- For Yes/No questions answer exactly "Yes" or "No"
- Keep answers concise (1-2 sentences max)
- If info not available write "To be provided"
- You MUST call fill_excel_cell for EVERY question — do not skip any"""

    user_message = f"""Fill each question below using the company profile.
Call fill_excel_cell for each one using the exact sheet/row/col provided.

COMPANY PROFILE:
{company_text}

QUESTIONS TO FILL ({len(questions)} total):
{fill_instructions}"""

    # Use fill_excel_cell only — no need for get_excel_structure
    fill_only_tools = [t for t in EXCEL_TOOLS if t["name"] == "fill_excel_cell"]

    messages = [{"role": "user", "content": user_message}]

    for _ in range(80):
        try:
            call_kwargs = dict(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=system_prompt,
                tools=fill_only_tools,
                tool_choice={"type": "any"},
                messages=messages,
            )
            response = client.messages.create(**call_kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(60)
                response = client.messages.create(**call_kwargs)
            else:
                return jsonify({"error": str(e)}), 500

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for tool_block in tool_blocks:
                result = execute_excel_tool(tool_block.name, tool_block.input, q_wb)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result
                })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
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


@app.route("/fetch-coralogix-kb", methods=["POST"])
@login_required
def fetch_coralogix_kb():
    """Fetch key Coralogix pages from the web and save to the Knowledge Base."""
    import concurrent.futures

    CORALOGIX_KEY_URLS = [
        "https://coralogix.com/about/",
        "https://coralogix.com/privacy-policy/",
        "https://coralogix.com/docs/security-and-compliance/",
        "https://coralogix.com/docs/soc-2/",
        "https://coralogix.com/docs/gdpr/",
        "https://coralogix.com/docs/iso-27001/",
        "https://coralogix.com/docs/data-security/",
        "https://coralogix.com/docs/privacy-and-security/",
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_cx_fetch, CORALOGIX_KEY_URLS))

    chunks = []
    fetched = 0
    for url, content in zip(CORALOGIX_KEY_URLS, results):
        if content.strip():
            chunks.append(f"===== {url} =====\n{content}\n")
            fetched += 1

    if not chunks:
        return jsonify({"error": "Could not fetch any Coralogix pages. Check your internet connection."}), 500

    save_path = os.path.join(KB_DIR, "coralogix_web_data.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

    return jsonify({"status": f"✅ Fetched {fetched} Coralogix pages", "pages": fetched})


# ── Knowledge Base ─────────────────────────────────────────────────────────────
# Use /data (Railway Volume) if available, otherwise fall back to local folder
KB_DIR = "/data/knowledge_base" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "knowledge_base")
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
@limiter.limit("50 per hour")
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


POLICY_FOLDER_ID = "1vJMCHwGEk2Ox5NZQcJEF3GhS9kN16Rje"


@app.route("/grc-policy-test", methods=["GET"])
@login_required
def grc_policy_test():
    """Diagnostic: test Drive connection and list files."""
    import json
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        return jsonify({"status": "ERROR", "reason": "GOOGLE_SERVICE_ACCOUNT_JSON env var is NOT set"}), 500
    try:
        sa_info = json.loads(sa_json)
        client_email = sa_info.get("client_email", "unknown")
    except Exception as e:
        return jsonify({"status": "ERROR", "reason": f"JSON parse failed: {e}"}), 500
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        results = service.files().list(
            q=f"'{POLICY_FOLDER_ID}' in parents and trashed=false",
            fields="files(id, name, modifiedTime)",
            pageSize=20,
        ).execute()
        files = results.get("files", [])
        return jsonify({
            "status": "OK",
            "service_account": client_email,
            "folder_id": POLICY_FOLDER_ID,
            "files_found": len(files),
            "files": [{"name": f["name"], "modified": f.get("modifiedTime","")[:10]} for f in files]
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "service_account": client_email, "reason": str(e)}), 500


@app.route("/grc-policy", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def grc_policy():
    """SOC 2 Policy Documents — reads real Google Drive folder and generates evidence report."""
    import json, datetime

    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        return jsonify({"error": "⚠️ GOOGLE_SERVICE_ACCOUNT_JSON not set in environment variables. See setup instructions."}), 500

    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        sa_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        # List all files (and subfolders) in the policy folder
        results = service.files().list(
            q=f"'{POLICY_FOLDER_ID}' in parents and trashed=false",
            fields="files(id, name, modifiedTime, mimeType, webViewLink)",
            orderBy="name",
            pageSize=100,
        ).execute()

        files = results.get("files", [])

    except Exception as e:
        return jsonify({"error": f"Google Drive error: {str(e)}"}), 500

    today = datetime.date.today().isoformat()

    if not files:
        return jsonify({
            "error": f"❌ No files found in the Drive folder.\n\nMost likely cause: the folder was not shared with the service account.\n\nPlease share the folder with:\ngrc-agent@coralogix-grc.iam.gserviceaccount.com\n\nThen try again."
        }), 500
    else:
        lines = []
        files_with_links = []
        for f in files:
            name     = f.get("name", "Unnamed")
            modified = f.get("modifiedTime", "")[:10]
            mime     = f.get("mimeType", "")
            fid      = f.get("id", "")
            link     = f.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view"
            ftype    = "folder" if mime == "application/vnd.google-apps.folder" else "file"
            lines.append(f"- {name}  |  last modified: {modified}  |  type: {ftype}")
            files_with_links.append({"name": name, "modified": modified, "link": link, "type": ftype})
        file_list = "\n".join(lines)

    user_message = f"""You are a SOC 2 GRC evidence collection agent auditing policy documents.

Control: _6_21 — Policy Documents
Audit date: {today}

Files found in the Google Drive policy folder:
{file_list}

Generate a structured, auditor-ready evidence report with these sections:

1. SUMMARY
   Total files found, date range of last modifications.

2. POLICY INVENTORY
   List every file with its last-modified date and a staleness flag:
   - UP TO DATE (modified within 12 months)
   - NEEDS REVIEW (modified 12-18 months ago)
   - OVERDUE (not modified in over 18 months)

3. COVERAGE CHECK
   Check whether these expected policy types appear in the list (by name):
   Information Security Policy, Access Control Policy, Incident Response Policy,
   Change Management Policy, Business Continuity / DR Policy, Acceptable Use Policy,
   Vendor Management Policy, Data Classification Policy, Risk Assessment Policy.
   For each: FOUND or MISSING.

4. COMPLIANCE STATUS
   Overall: PASS / NEEDS ATTENTION / FAIL with a one-line justification.

5. AUDITOR SUMMARY
   One paragraph suitable for inclusion in an audit evidence package.

Use plain text and clear section headers. No markdown asterisks or bold."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{"role": "user", "content": user_message}],
        )
        result = next((b.text for b in response.content if hasattr(b, "text")), "No result generated.")
        return jsonify({"result": result, "files": files_with_links})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


HIBOB_API_URL = "https://api.hibob.com/v1"


@app.route("/grc-hibob-test", methods=["GET"])
@login_required
def grc_hibob_test():
    """Diagnostic: test HiBob connection and return employee count."""
    import requests as req

    token = os.getenv("HIBOB_SERVICE_TOKEN")
    if not token:
        return jsonify({"status": "ERROR", "reason": "HIBOB_SERVICE_TOKEN env var is NOT set"}), 500

    try:
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }
        resp = req.get(f"{HIBOB_API_URL}/people", headers=headers, timeout=30)

        if resp.status_code == 401:
            return jsonify({"status": "ERROR", "reason": "401 Unauthorized — check HIBOB_SERVICE_TOKEN value"}), 500
        if resp.status_code != 200:
            return jsonify({"status": "ERROR", "reason": f"HiBob API {resp.status_code}: {resp.text[:300]}"}), 500

        data = resp.json()
        employees = data.get("employees", [])
        from collections import Counter
        sites = Counter(e.get("work", {}).get("site", "Unknown") for e in employees)
        return jsonify({
            "status": "OK",
            "employees_found": len(employees),
            "sites": dict(sites),
            "sample": [{"name": e.get("displayName"), "site": e.get("work", {}).get("site")} for e in employees[:5]],
        })
    except Exception as e:
        return jsonify({"status": "ERROR", "reason": str(e)}), 500


@app.route("/grc-hibob", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def grc_hibob():
    """SOC 2 Active Employee List (_10) — live data from HiBob."""
    import requests as req
    from collections import Counter
    import datetime

    token = os.getenv("HIBOB_SERVICE_TOKEN")
    if not token:
        return jsonify({"error": "⚠️ HIBOB_SERVICE_TOKEN is not set. Add it to Railway environment variables."}), 500

    try:
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }
        resp = req.get(f"{HIBOB_API_URL}/people", headers=headers, timeout=30)

        if resp.status_code == 401:
            return jsonify({"error": "HiBob authentication failed (401 Unauthorized). Check your HIBOB_SERVICE_TOKEN."}), 500
        if resp.status_code != 200:
            return jsonify({"error": f"HiBob API returned {resp.status_code}: {resp.text[:300]}"}), 500

        data = resp.json()
        employees = data.get("employees", [])

        if not employees:
            return jsonify({"error": "No employees returned by HiBob. Check API permissions for the service user."}), 404

    except Exception as e:
        return jsonify({"error": f"HiBob connection error: {str(e)}"}), 500

    # --- Build structured data ---
    site_counts   = Counter()
    dept_counts   = Counter()
    employee_rows = []

    for emp in employees:
        work = emp.get("work", {})
        site = work.get("site") or "Unknown"
        dept = work.get("department") or "Unknown"
        name = emp.get("displayName") or "Unknown"
        title      = work.get("title") or ""
        start_date = (work.get("startDate") or "")[:10]

        site_counts[site] += 1
        dept_counts[dept] += 1
        employee_rows.append({
            "name":      name,
            "site":      site,
            "dept":      dept,
            "title":     title,
            "startDate": start_date or "N/A",
        })

    today       = datetime.date.today().isoformat()
    total       = len(employee_rows)
    site_table  = "\n".join(
        f"  - {s}: {c} ({round(c/total*100)}%)"
        for s, c in sorted(site_counts.items(), key=lambda x: -x[1])
    )
    dept_table  = "\n".join(
        f"  - {d}: {c}"
        for d, c in sorted(dept_counts.items(), key=lambda x: -x[1])[:12]
    )
    sample_rows = "\n".join(
        f"  {i+1}. {e['name']} | {e['site']} | {e['dept']} | {e['title']} | Since {e['startDate']}"
        for i, e in enumerate(employee_rows[:30])
    )

    user_message = f"""You are a SOC 2 GRC evidence collection agent.

Control: _10 — Active Employee List
Data source: HiBob HRIS (live API pull)
Audit date: {today}

LIVE DATA:
Total active employees: {total}

Headcount by site/entity:
{site_table}

Headcount by department (top 12):
{dept_table}

Sample employee records (first 30 of {total}):
{sample_rows}

Generate a structured, auditor-ready evidence report with these sections:

1. SUMMARY
   Total headcount, number of distinct entities/sites, data pull date.

2. HEADCOUNT BY ENTITY
   Table of each site with employee count and % of total workforce.

3. COMPLIANCE RELEVANCE
   Confirm this employee population is the correct scope for:
   - Access provisioning / deprovisioning reviews
   - Security awareness training completion tracking
   - Background check coverage
   Flag any entity with fewer than 5 employees as potentially needing investigation.

4. AUDITOR ATTESTATION
   A concise paragraph confirming the population was extracted live from HiBob
   and represents the complete active workforce as of the audit date.

5. OVERALL STATUS: clearly state ✅ PASS, ⚠️ NEEDS ATTENTION, or ❌ FAIL

Use plain text, no markdown asterisks, clear section headers."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1400,
            messages=[{"role": "user", "content": user_message}],
        )
        result = next((b.text for b in response.content if hasattr(b, "text")), "No result generated.")
        return jsonify({
            "result": result,
            "employee_count": total,
            "sites": [{"site": s, "count": c} for s, c in sorted(site_counts.items(), key=lambda x: -x[1])],
        })
    except Exception as e:
        return jsonify({"error": f"Claude error: {str(e)}"}), 500


@app.route("/grc-agent", methods=["POST"])
@login_required
@limiter.limit("50 per hour")
def grc_agent():
    """SOC 2 GRC evidence collection agent."""
    data = request.get_json()
    control_id = data.get("controlId", "")
    title      = data.get("title", "")
    task       = data.get("task", "")

    if not task:
        return jsonify({"error": "No task provided"}), 400

    system_prompt = """You are a SOC 2 GRC (Governance, Risk & Compliance) evidence collection agent.
Your role is to collect and summarise compliance evidence for SOC 2 Type II audits.

For each control produce a structured evidence report with:
1. Control ID and category
2. Evidence collected (what was pulled from the relevant systems)
3. Compliance status — clearly marked as ✅ PASS, ⚠️ NEEDS ATTENTION, or ❌ FAIL
4. Specific findings with realistic dates, counts, and names
5. Any gaps or remediation actions required
6. A one-paragraph auditor-ready summary

Format the report with clear section headers and plain text (no markdown bold/asterisks).
Write as if you have just queried the live systems and are reporting findings in real time."""

    user_message = (
        f"Collect SOC 2 evidence for the following control.\n\n"
        f"Control: {control_id} — {title}\n"
        f"Task: {task}\n\n"
        f"Generate a realistic, detailed evidence report an auditor would accept."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        result = next((b.text for b in response.content if hasattr(b, "text")), "No result generated.")
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Contract KPI ──────────────────────────────────────────────────────────────
import uuid
from datetime import date

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
CONTRACTS_FILE = os.path.join(_DATA_DIR, "contracts.json")

def load_contracts():
    if not os.path.exists(CONTRACTS_FILE):
        return []
    with open(CONTRACTS_FILE, "r") as f:
        return json.load(f)

def save_contracts(contracts):
    with open(CONTRACTS_FILE, "w") as f:
        json.dump(contracts, f, indent=2)


@app.route("/add-contract", methods=["POST"])
@login_required
def add_contract():
    try:
        client     = request.form.get("client", "").strip()
        start_date = request.form.get("start_date", "")
        sign_date  = request.form.get("sign_date", "")
        if not client or not start_date or not sign_date:
            return jsonify({"error": "Missing fields"}), 400

        start = date.fromisoformat(start_date)
        sign  = date.fromisoformat(sign_date)
        if sign < start:
            return jsonify({"error": "Sign date before start date"}), 400

        duration = (sign - start).days
        contract_id = str(uuid.uuid4())

        # Save file if provided
        has_file = False
        file = request.files.get("file")
        if file and file.filename:
            file_path = os.path.join(os.path.dirname(__file__), f"contract_{contract_id}.bin")
            file.save(file_path)
            has_file = True

        contracts = load_contracts()
        contracts.append({
            "id": contract_id,
            "client": client,
            "start_date": start_date,
            "sign_date": sign_date,
            "duration_days": duration,
            "has_file": has_file,
        })
        save_contracts(contracts)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/list-contracts", methods=["GET"])
@login_required
def list_contracts():
    return jsonify({"contracts": load_contracts()})


@app.route("/delete-contract", methods=["POST"])
@login_required
def delete_contract():
    contract_id = request.json.get("id")
    contracts = [c for c in load_contracts() if c["id"] != contract_id]
    save_contracts(contracts)
    file_path = os.path.join(os.path.dirname(__file__), f"contract_{contract_id}.bin")
    if os.path.exists(file_path):
        os.remove(file_path)
    return jsonify({"ok": True})


@app.route("/analyze-contract", methods=["POST"])
@login_required
def analyze_contract():
    try:
        contract_id = request.json.get("id")
        contracts = load_contracts()
        contract = next((c for c in contracts if c["id"] == contract_id), None)
        if not contract:
            return jsonify({"error": "Contract not found"}), 404

        file_path = os.path.join(os.path.dirname(__file__), f"contract_{contract_id}.bin")
        if not os.path.exists(file_path):
            return jsonify({"error": "No file found for this contract"}), 404

        with open(file_path, "rb") as f:
            content = f.read()

        # Try to decode as text
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="replace")

        # Truncate if too long
        text = text[:12000]

        msg = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Analyze this contract for client "{contract['client']}".
Duration from start to signing: {contract['duration_days']} days.

Please provide:
1. Key obligations and commitments
2. Important dates or deadlines mentioned
3. Risk factors or red flags
4. What can be learned / improved for future contracts
5. Overall assessment

Contract content:
{text}"""
            }]
        )
        return jsonify({"insights": msg.content[0].text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
