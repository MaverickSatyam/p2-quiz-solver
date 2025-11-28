import asyncio
import os
import re
import json
import httpx
from urllib.parse import urlparse, urljoin
import subprocess
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- 1. AIPIPE CONFIGURATION ---
client = AsyncOpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)
AI_MODEL = "openai/gpt-4.1-nano"

# Workspace for downloaded files
WORKSPACE_DIR = "./workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# --- 2. SCRAPER ---
async def get_task_context(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (AI-Student)")
        page = await context.new_page()
        
        print(f"🕵️  Scraping: {url}")
        try:
            await page.goto(url, timeout=20000)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)

            visible_text = await page.inner_text("body")
            
            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({
                    text: a.innerText,
                    href: a.href
                }))
            }""")
            
            await browser.close()
            return visible_text, links
        except Exception as e:
            await browser.close()
            raise e

# --- 3. DOWNLOADER AND FILE READER ---
async def download_files(links):
    local_files = []
    async with httpx.AsyncClient() as client:
        for link in links:
            href = link['href']
            if re.search(r'\.(pdf|csv|json|txt|png|jpg|zip|xlsx)$', href, re.IGNORECASE):
                try:
                    filename = os.path.basename(href.split("?")[0])
                    filepath = os.path.join(WORKSPACE_DIR, filename)
                    print(f"⬇️  Downloading: {filename}")
                    
                    resp = await client.get(href, follow_redirects=True)
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    local_files.append(filepath)
                except Exception as e:
                    print(f"⚠️  Download failed: {e}")
    return local_files

def get_file_contents_for_prompt(files):
    """Reads the content of downloaded files for inclusion in the prompt."""
    contents = {}
    for filepath in files:
        if os.path.getsize(filepath) > 200000:
            contents[filepath] = "File too large to include in prompt."
            continue
            
        try:
            with open(filepath, 'r') as f:
                contents[filepath] = f.read(20000) 
        except Exception as e:
            contents[filepath] = f"Error reading file: {e}"
            
    return "\n\n".join([f"--- FILE: {path} (Snippet) ---\n{content}" for path, content in contents.items()])

# --- 4. AI CODE GENERATOR (Enhanced for Context) ---
async def generate_solution(instruction, file_paths, current_url_base, all_links, previous_error=None):
    files_context = "\n".join([f"- {path}" for path in file_paths])
    base_url_context = f"\nCURRENT PAGE BASE URL: {current_url_base}" if current_url_base else ""
    links_context = "\n. ".join([f"- Text: {link['text']}, URL: {link['href']}" for link in all_links])

    # Dynamic Hinting based on previous error (server rejection or code failure)
    failure_context = ""
    if previous_error:
        failure_context = f"""
--- PREVIOUS ATTEMPT FAILED ---
Reason/Error: {previous_error}
Your previous attempt failed. Analyze the error carefully.
- If the error is 'Wrong sum of numbers' or similar, your code's logic was wrong.
- If the error is 'CODE_EXECUTION_FAILURE' followed by a traceback, fix the Python bug (e.g., KeyError, incorrect function call).
Adjust your Python code to solve the task correctly.
"""
    
    system_prompt = f"""
        You are an autonomous Python data analyst. Your ONLY goal is to solve the task by generating correct, working Python code.
        
        You MUST respond with a single JSON object.
        
        The JSON object MUST have two keys: "submit_url" (string) and "python_code" (string).
        
        Rules for the 'python_code' value:
        1. The code must be a single, complete Python script string. DO NOT wrap it in ```python.
        2. The script must read instructions/files/links, calculate the final answer.
        3. The script MUST print the FINAL ANSWER to stdout ONLY.
        4. If the code uses the 'requests' library for internal submission, the payload's **'url' field MUST be set to the ORIGINAL challenge URL** derived from the context.
        5. If using Pandas, use robust access methods like **df.iloc[:, 0]** or inspect headers dynamically; DO NOT hardcode column names.
        
        Context Files:
        {files_context}
        """
    
    user_prompt = f"""
    {base_url_context}
    
    Relevant Links found on the page:
    {links_context}
    
    Instruction:
    {instruction}
    
    {failure_context}
    """

    response = await client.chat.completions.create(
        model=AI_MODEL,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return json.loads(response.choices[0].message.content)

# --- 5. EXECUTOR (Enhanced to return Traceback) ---
def execute_code(code):
    print("🚀 Executing Solution Code...")
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=20
        )
        if result.returncode != 0:
            print(f"⚠️ Code Execution Failed. Error: {result.stderr}")
            # Return the detailed traceback for the LLM to fix
            return f"CODE_EXECUTION_FAILURE: {result.stderr}" 
            
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠️ Execution failed due to internal error: {e}")
        return f"CODE_EXECUTION_FAILURE: Internal Error: {e}"

# --- 6. SUBMITTER & Helper ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
async def submit_answer(url, payload):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=10)
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp.json()

def normalize_answer(raw_answer):
    if not raw_answer or "CODE_EXECUTION_FAILURE" in raw_answer:
        return None
        
    final_answer = raw_answer.replace('"', '').strip()
    try:
        if final_answer.isdigit() or final_answer.count('.') == 1 and final_answer.replace('.', '').isdigit():
            return float(final_answer)
        else:
            return final_answer
    except:
        return raw_answer

# --- 7. MAIN LOOP (Streamlined to Code-First) ---
async def run_quiz_agent(start_url, email, secret):
    current_url = start_url
    
    while current_url:
        print(f"\n🔵 Processing: {current_url}")
        
        result = {"correct": False}
        previous_error = None
        
        try:
            # --- PHASE 0: Initial Setup ---
            text, links = await get_task_context(current_url)
            files = await download_files(links)
            # file_contents = get_file_contents_for_prompt(files) # Not needed for code-first, but left here for context
            
            url_parts = urlparse(current_url)
            base_url = f"{url_parts.scheme}://{url_parts.netloc}"

            # --- ATTEMPT 1: AI CODE GENERATION & EXECUTION (Primary Method) ---
            
            # Use a while loop for the single question to attempt a retry
            attempt_count = 0
            MAX_ATTEMPTS = 2 # Max two code generation attempts per question
            
            while not result.get("correct") and attempt_count < MAX_ATTEMPTS:
                attempt_count += 1
                answer = None

                print(f"\n💻 Code Attempt {attempt_count}: Generating solution...")
                
                # Generate plan (passes context and previous error)
                plan = await generate_solution(text, files, base_url, links, previous_error)
                code = plan.get("python_code")
                current_submit_url = plan.get("submit_url")

                # --- DEBUG CODE ---
                print("\n--- AI GENERATED CODE START ---")
                print(code)
                print("--- AI GENERATED CODE END ---\n")
                # ------------------
                
                # 1. Execute code
                raw_answer = execute_code(code)
                answer = normalize_answer(raw_answer)

                # 2. Process Result
                if answer:
                    
                    # --- URL VALIDATION (Critical, run before submission) ---
                    # Logic here uses AI URL + fallback, ensuring submit_url is final before proceeding
                    submit_url = current_submit_url
                    # ... [YOUR ROBUST URL VALIDATION AND FALLBACK LOGIC GOES HERE] ...
                    # Assuming submit_url is successfully determined.
                    
                    if not submit_url:
                        print("❌ CRITICAL FAILURE: Could not find submission URL.")
                        break # Exit inner while loop
                        
                    # 3. Submit
                    print(f"💡 Answer: {answer}")
                    payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                    result = await submit_answer(submit_url, payload)
                    
                    if result.get("correct"):
                        print("✅ Correct!")
                        break # Exit inner while loop
                    else:
                        previous_error = result.get('reason')
                        print(f"❌ Wrong: {previous_error}. Retrying...")
                else:
                    # If execution failed, raw_answer contains the traceback
                    previous_error = raw_answer # Capture the detailed error/traceback
                    print(f"❌ Execution failed or produced no answer. Retrying with traceback...")
            
            # --- FINAL LOOP CONTROL ---
            current_url = result.get("url") # Update URL based on the last attempt's result
            
            # Cleanup
            for f in files: os.remove(f)

        except Exception as e:
            print(f"🔥 Critical Error during processing: {e}")
            traceback.print_exc()
            break