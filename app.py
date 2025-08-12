# app.py
from __future__ import annotations

import os
import re
import sys
import json
import subprocess
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from google import genai

# ---------- Config (env-driven) ----------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Crash early & clearly if the key isn't there
        raise RuntimeError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)

# ---------- FastAPI app ----------
app = FastAPI(title="TDS Project – Multi-step LLM Runner")

# CORS (open for simplicity; tighten if you have a frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _strip_code_fences(txt: str) -> str:
    # Remove ```lang and ``` fences; keep inner code only
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", txt)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()

# ---------- LLM stages ----------
def task_breakdown(task: str) -> str:
    """Stage 1: Ask Gemini to break down the user's task."""
    client = get_gemini_client()
    prompt_path = os.path.join("prompts", "b-2.txt")
    task_breakdown_prompt = _read_text(prompt_path)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[task, task_breakdown_prompt],
    )
    text = resp.text or ""
    _write_text("broken_tasks.txt", text)
    return text

def write_code(_: str) -> str:
    """Stage 2: Ask Gemini to write Python code for the broken-down steps."""
    tasks = _read_text("broken_tasks.txt")
    client = get_gemini_client()

    prompt_path = os.path.join("prompts", "b-3.txt")
    task_write_prompt = _read_text(prompt_path)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[tasks, task_write_prompt],
    )
    code = _strip_code_fences(resp.text or "")
    _write_text("temp_script.py", code)
    return code

def iterate_code(code: str, max_iterations: int = 20) -> str:
    """Stage 3: Iterate until the script runs without stderr, using b-4.txt prompt."""
    client = get_gemini_client()

    for i in range(max_iterations):
        run = subprocess.run(
            [sys.executable, "-I", "temp_script.py"],
            capture_output=True,
            text=True,
        )
        if run.returncode == 0 and not run.stderr.strip():
            # Success
            break

        # Ask the model to fix code using stderr as feedback
        fix_prompt_path = os.path.join("prompts", "b-4.txt")
        fix_code_prompt = _read_text(fix_prompt_path) + "\n\n" + (run.stderr or "")

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[code, fix_code_prompt],
        )
        code = _strip_code_fences(resp.text or "")
        _write_text("temp_script.py", code)

    return code

# ---------- Routes ----------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Hello!"}

@app.post("/api/")
async def upload_file(request: Request):
    """
    Accept multipart/form-data with at least one file named 'questions.txt'.
    All other files are saved under ./attachments for the generated code to use.
    """
    try:
        form = await request.form()
        files: List = []
        questions_file = None

        os.makedirs("attachments", exist_ok=True)

        # Collect files; detect the required questions.txt
        for key in form:
            value = form[key]
            if hasattr(value, "filename"):  # UploadFile
                files.append(value)
                if value.filename == "questions.txt":
                    questions_file = value

        if not questions_file:
            return JSONResponse(status_code=400, content={"error": "questions.txt is required."})

        # Read the user's question text
        q_bytes = await questions_file.read()
        question_text = q_bytes.decode("utf-8", errors="ignore")

        # Save other files as attachments for the generated script
        for f in files:
            if f.filename != "questions.txt":
                fp = os.path.join("attachments", f.filename)
                # read file content now (stream pointer is at start)
                content = await f.read()
                with open(fp, "wb") as out:
                    out.write(content)

        # Run the LLM pipeline
        breakdown = task_breakdown(question_text)
        code = write_code(breakdown)
        final_code = iterate_code(code)

        # Final run to produce the answer
        final_run = subprocess.run(
            [sys.executable, "-I", "temp_script.py"],
            capture_output=True,
            text=True,
        )
        output_text = final_run.stdout.strip()

        # Try to return JSON if it's valid JSON; else plain text
        try:
            return JSONResponse(content=json.loads(output_text))
        except json.JSONDecodeError:
            return PlainTextResponse(output_text)

    except Exception as e:
        # Return a structured error
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # Local dev only; on Railway Procfile will launch uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
