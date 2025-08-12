# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
# ]
# ///

from fastapi import FastAPI, Request  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os
import re
import subprocess

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)
def task_breakdown(task:str):
    """Breaks down a task into smaller programmable steps using Google GenAI."""
    client = genai.Client(api_key="asfd")

    task_breakdown_file = os.path.join('prompts', "b-2.txt")
    with open(task_breakdown_file, 'r') as f:
        task_breakdown_prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[task,task_breakdown_prompt],
    )

    with open("broken_tasks.txt", "w") as f:
        f.write(response.text)

    return response.text

def write_code(breakdown:str):
    """Read the small broken down steps and return this file to the LLM which then creates the code to run."""
    with open("broken_tasks.txt", "r") as f:
        tasks = f.read()
        
    client = genai.Client(api_key="got you")

    file = os.path.join('prompts', "b-3.txt")
    with open(file, 'r') as f:
        task_write_prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[tasks, task_write_prompt],
    )

    txt = response.text
    # Remove code fences like ```python, ``` and variations
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", txt)
    cleaned = cleaned.replace("```", "")

    with open("temp_script.py", "w") as f:
        f.write(cleaned)

    return cleaned

def iterate_code(code: str):
    """Fix code until it runs without errors using b-4.txt prompt."""
    client = genai.Client(api_key="adfs")


    max_iterations = 20
    for _ in range(max_iterations):
        result = subprocess.run(["uv", "run", "temp_script.py"], capture_output=True, text=True)

        if not result.stderr:  # No errors
            break

        fix_code_file = os.path.join('prompts', "b-4.txt")
        with open(fix_code_file, 'r') as f:
            fix_code_prompt = f.read()

        fix_code_prompt = fix_code_prompt + "\n" + result.stderr
        print(f"Iteration {_+1}: Fixing code with prompt:\n{fix_code_prompt}")
        # Get model suggestion
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[code, fix_code_prompt],
        )

        # Extract text properly depending on SDK
        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", response.text)
        cleaned = cleaned.replace("```", "")
        code = cleaned  # Adjust if API returns structured parts

        # Save fixed code
        with open("temp_script.py", "w") as f:
            f.write(code)

    return code

@app.get("/")
async def root():
    return {"message": "Hello!"}

# create a post endpoint that processes this curl request `curl -X POST "http://127.0.0.1:8000/api/" -F "file=@question.txt"`

# Accept both a text file and a CSV file in the request

# Accept any number of files, with questions.txt always present


# Accept any files regardless of field name
@app.post("/api/")
async def upload_file(request: Request):
    try:
        form = await request.form()
        files = []
        questions_file = None
        os.makedirs("attachments", exist_ok=True)

        # Iterate over all form fields
        for key in form:
            value = form[key]
            if hasattr(value, "filename"):
                files.append(value)
                if value.filename == "questions.txt":
                    questions_file = value

        if not questions_file:
            return JSONResponse(status_code=400, content={"error": "questions.txt is required."})

        content = await questions_file.read()
        text = content.decode("utf-8")
        breakdown = task_breakdown(text)

        # Save all files except questions.txt
        for f in files:
            if f.filename != "questions.txt":
                file_path = os.path.join("attachments", f.filename)
                with open(file_path, "wb") as out:
                    out.write(await f.read())

        final_code = iterate_code(write_code(breakdown))
        json_ans = subprocess.run(["uv", "run", "temp_script.py"], capture_output=True, text=True).stdout

        print(json_ans)
        return json_ans
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
