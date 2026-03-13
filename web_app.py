import os
from typing import Literal, TypedDict

import ollama
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pypdf import PdfReader


DocumentType = Literal["invoice", "purchase_order"]


class ClassificationResult(TypedDict):
    type: DocumentType
    confidence: float
    explanation: str
    # Holds the structured fields extracted from the document
    # For invoices:
    #   - invoice_number
    #   - customer
    #   - invoice_date
    # For purchase orders:
    #   - order_number
    #   - customer
    #   - date
    fields: dict[str, str]


SYSTEM_PROMPT = """You are a document classifier for the company Red Thread.

You will be given the text content of a single document that was sent to Red Thread.
Your job is to decide if the document is:
- an invoice
- or a purchase order

Rules:
- Only consider INVOICES or PURCHASE ORDERS (POs) related to Red Thread or its business context.
- Carefully read for typical signals: headings, line items, totals, "Invoice", "Purchase Order", "PO #", payment terms, billing vs shipping info, etc.
- If it clearly mixes both, classify based on the PRIMARY purpose of the document.
- If you are uncertain, choose the more likely type but reduce your confidence.

Return your answer strictly as a JSON object with this schema:
{
  "type": "invoice" | "purchase_order",
  "confidence": number between 0 and 1,
  "explanation": "short natural language explanation"
}

Do not include any extra text outside the JSON.
"""


EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant for Red Thread.

You will receive:
- the full text of a single business document sent to or from Red Thread
- and a label telling you whether it is an "invoice" or a "purchase_order".

Your job:
- If the document is an INVOICE, extract:
  - "invoice_number": the main invoice number (Invoice #, Invoice No, etc.)
  - "customer": the customer company that the invoice is billed TO (who receives invoice from Red Thread)
  - "invoice_date": the invoice date

- If the document is a PURCHASE ORDER, extract:
  - "order_number": the purchase order number (PO #, Order #, etc.)
  - "customer": the customer company sending the PO to Red Thread (generally under a TO / Bill To / Sold To section)
  - "date": the PO date

Rules:
- Prefer clearly labeled fields ("Invoice #", "PO #", "Order Number", "Invoice Date", "Date", "Customer", "Bill To", "Sold To") over guessing.
- If a field truly cannot be found, set its value to an empty string "".
- Keep values concise and human-readable (no surrounding labels or extra commentary).

Return your answer strictly as JSON with this schema:
{
  "fields": {
    "invoice_number" | "order_number": "string",
    "customer": "string",
    "invoice_date" | "date": "string"
  }
}

Do not include any extra text outside the JSON.
"""


def extract_text_from_pdf_bytes(data: bytes, max_pages: int = 5) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    pages = reader.pages[: max_pages]
    parts: list[str] = []
    for page in pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n\n".join(parts)


def extract_text_from_upload(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf_bytes(data)
    # Treat as text
    return data.decode("utf-8", errors="ignore")


def call_ollama_classifier(text: str, model: str = "gpt-oss:20b") -> ClassificationResult:
    max_chars = 12000
    snippet = text[:max_chars]

    user_prompt = f"""Classify the following document as an invoice or a purchase order for Red Thread.

Document content:
\"\"\"{snippet}\"\"\""""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response["message"]["content"]

    content_stripped = content.strip()
    if content_stripped.startswith("```"):
        content_stripped = content_stripped.strip("`")
    start = content_stripped.find("{")
    end = content_stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = content_stripped[start : end + 1]
    else:
        json_str = content_stripped

    import json

    data = json.loads(json_str)

    doc_type = data.get("type", "").strip().lower()
    if doc_type not in ("invoice", "purchase_order"):
        raise ValueError(f"Unexpected type from model: {doc_type!r}")

    confidence = float(data.get("confidence", 0.0))
    explanation = str(data.get("explanation", "")).strip()

    # `fields` will be filled in later by the extraction call
    return {
        "type": doc_type,  # type: ignore[typeddict-item]
        "confidence": confidence,
        "explanation": explanation,
        "fields": {},
    }


def call_ollama_extractor(
    text: str, doc_type: DocumentType, model: str = "gpt-oss:20b"
) -> dict[str, str]:
    """
    Ask the model to pull out the key business fields
    based on whether the document is an invoice or purchase order.
    """
    max_chars = 12000
    snippet = text[:max_chars]

    user_prompt = f"""Extract the key fields for this {doc_type} document.

Document content:
\"\"\"{snippet}\"\"\""""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response["message"]["content"]
    content_stripped = content.strip()
    if content_stripped.startswith("```"):
        content_stripped = content_stripped.strip("`")
    start = content_stripped.find("{")
    end = content_stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = content_stripped[start : end + 1]
    else:
        json_str = content_stripped

    import json

    data = json.loads(json_str)
    fields = data.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("Extractor response did not contain a 'fields' object")
    # Normalize keys to simple strings
    return {str(k): str(v) for k, v in fields.items()}


app = FastAPI(title="Red Thread Document Classifier")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Red Thread Document Classifier</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #5a0c1f, #1a0a0f 55%, #0b0608);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: #7a1124; /* maroon-ish red */
      border-radius: 20px;
      padding: 40px 48px;
      box-shadow: 0 22px 55px rgba(0, 0, 0, 0.55);
      max-width: 640px;
      width: 100%;
      box-sizing: border-box;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 28px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    p.subtitle {
      margin: 0 0 28px;
      font-size: 15px;
      opacity: 0.9;
    }
    .upload-label {
      display: inline-block;
      margin-bottom: 18px;
      font-size: 14px;
      opacity: 0.9;
    }
    input[type="file"] {
      display: block;
      margin: 0 auto 24px;
      color: #fff;
    }
    button {
      background: #ff4d4d;
      color: #fff;
      border: none;
      border-radius: 999px;
      padding: 10px 26px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.55);
      transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.1s ease;
    }
    button:hover {
      background: #ff6b6b;
      transform: translateY(-1px);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5);
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
      box-shadow: none;
      transform: none;
    }
    .status {
      margin-top: 14px;
      font-size: 13px;
      opacity: 0.9;
    }
    .result {
      margin-top: 22px;
      font-size: 17px;
      font-weight: 600;
    }
    .result-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
      text-align: left;
    }
    .result-table th,
    .result-table td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.18);
    }
    .result-table th {
      font-weight: 600;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.06em;
      opacity: 0.85;
    }
    .spinner {
      margin: 10px auto 0;
      border: 3px solid rgba(255,255,255,0.25);
      border-top: 3px solid #fff;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      animation: spin 0.7s linear infinite;
      display: none;
    }
    .spinner.visible {
      display: block;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Red Thread</h1>
    <p class="subtitle">Smart classifier for invoices vs purchase orders.</p>
    <label class="upload-label" for="file">Upload a PDF or text document:</label>
    <input id="file" type="file" accept=".pdf,.txt" />
    <button id="submitBtn">Upload &amp; analyze</button>
    <div id="spinner" class="spinner"></div>
    <div id="status" class="status"></div>
    <div id="result" class="result"></div>
  </div>

  <script>
    const fileInput = document.getElementById("file");
    const submitBtn = document.getElementById("submitBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const spinner = document.getElementById("spinner");

    submitBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.textContent = "Please choose a file first.";
        resultEl.textContent = "";
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      submitBtn.disabled = true;
      statusEl.textContent = "Analyzing document, please wait...";
      resultEl.textContent = "";
      spinner.classList.add("visible");

      try {
        const response = await fetch("/classify", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorText = await response.text();
          statusEl.textContent = "Error analyzing document.";
          resultEl.textContent = errorText || "Unknown error.";
        } else {
          const data = await response.json();
          statusEl.textContent = "Analysis complete.";

          const docLabel =
            data.type === "invoice"
              ? "Invoice"
              : data.type === "purchase_order"
              ? "Purchase Order"
              : "Unknown document";

          const fields = data.fields || {};

          const rows = [];
          if (data.type === "invoice") {
            rows.push(["Invoice #", fields.invoice_number || ""]);
            rows.push(["Customer", fields.customer || ""]);
            rows.push(["Invoice date", fields.invoice_date || ""]);
          } else if (data.type === "purchase_order") {
            rows.push(["Order #", fields.order_number || ""]);
            rows.push(["Customer", fields.customer || ""]);
            rows.push(["Date", fields.date || ""]);
          } else {
            for (const [k, v] of Object.entries(fields)) {
              rows.push([k, v]);
            }
          }

          const rowsHtml = rows
            .map(
              ([label, value]) =>
                `<tr><td>${label}</td><td>${value || "<span style='opacity:0.6'>(not found)</span>"}</td></tr>`
            )
            .join("");

          resultEl.innerHTML = `
            <div>${docLabel} detected</div>
            <table class="result-table">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                ${rowsHtml}
              </tbody>
            </table>
          `;
        }
      } catch (err) {
        statusEl.textContent = "Failed to reach server.";
        resultEl.textContent = String(err);
      } finally {
        submitBtn.disabled = false;
        spinner.classList.remove("visible");
      }
    });
  </script>
</body>
</html>
    """


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> ClassificationResult:
    try:
        content = await file.read()
        text = extract_text_from_upload(file.filename, content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable.")

        # First classify the document (invoice vs purchase order)
        result = call_ollama_classifier(text)

        # Then extract key business fields based on the inferred type
        try:
            extracted_fields = call_ollama_extractor(text, result["type"])
        except Exception:
            # If extraction fails, still return the classification
            extracted_fields = {}

        result["fields"] = extracted_fields
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    url = f"http://localhost:{port}/"
    print(f"Red Thread Document Classifier running at {url}")
    print("Press Ctrl+C to stop.")
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=True)

