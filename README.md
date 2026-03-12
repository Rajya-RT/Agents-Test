## Red Thread Document Classifier (Web UI)

This project exposes your local **Ollama** model as a small **web app** that determines whether an uploaded file is an **invoice** or a **purchase order** for Red Thread.

It supports:
- **PDF files** (first few pages are extracted)
- **Plain text files**

The backend lives in `web_app.py` and:
- extracts text from the uploaded file,
- sends it to your local Ollama server with a clear system prompt,
- and returns a JSON result with `type`, `confidence`, and `explanation`.

The frontend is a simple, red-themed HTML page served at `/` with:
- a file picker,
- an **Upload & analyze** button,
- a loading spinner while the document is being analyzed,
- and a final message: **"Invoice uploaded"** or **"PO uploaded"**.

### Prerequisites

- Python 3.10+ installed
- Ollama installed and running locally
- Your model (e.g. `gpt-oss:20b`) already pulled in Ollama, for example:

```bash
ollama pull gpt-oss:20b
```

Make sure the Ollama server is running (typically it starts automatically; on Windows you may start it from the Ollama app).

### Setup

From this project directory:

```bash
pip install -r requirements.txt
```

### Run the web server

From this project directory:

```bash
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

Then open this URL in a browser (on this machine or others on your network):

```text
http://<your-computer-ip>:8000/
```

For example, on the same machine you can go to:

```text
http://localhost:8000/
```

Upload a PDF or text file and click **Upload & analyze**.  
The page will show a loading indicator and then display **"Invoice uploaded"** or **"PO uploaded"**.
