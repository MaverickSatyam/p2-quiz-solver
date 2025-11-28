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
AI_MODEL = "openai/gpt-4o"  # Use high-quality model for coding tasks

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
            await page.wait_for_timeout(2000) # Wait for JS rendering

            visible_text = await page.inner_text("body")
            
            # Get all links for file downloads
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

# --- 3. DOWNLOADER ---
async def download_files(links):
    local_files = []
    async with httpx.AsyncClient() as client:
        for link in links:
            href = link['href']
            # Download if it looks like a data file
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

# --- 4. AI CODE GENERATOR ---
async def generate_solution(instruction, file_paths):
    files_context = "\n".join([f"- {path}" for path in file_paths])
    
    system_prompt = """
    You are an autonomous data analyst.
    1. READ the user instruction.
    2. WRITE a Python script to solve the task.
    3. OUTPUT a JSON object containing the code and the submission URL.
    
    The code must:
    - Print the FINAL ANSWER to stdout.
    - Handle errors gracefully.
    - Use pandas, numpy, pdfplumber, etc. as needed.
    """
    
    user_prompt = f"""
    Context Files:
    {files_context}
    
    Instruction:
    {instruction}
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

# --- 5. EXECUTOR ---
def execute_code(code):
    print("🚀 Executing Solution Code...")
    try:
        # Run python code in a subprocess
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=20
        )
        if result.returncode != 0:
            print(f"⚠️ Code Execution Failed. Error: {result.stderr}")
            # If code fails, return a specific failure string, NOT None
            return "CODE_EXECUTION_FAILED" 
            
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠️ Execution failed due to internal error: {e}")
        return "CODE_EXECUTION_FAILED"

# --- 6. SUBMITTER ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
async def submit_answer(url, payload):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=10)
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp.json()

# --- 7. MAIN LOOP ---
async def run_quiz_agent(start_url, email, secret):
    current_url = start_url
    
    while current_url:
        print(f"\n🔵 Processing: {current_url}")
        try:
            # Step A: Get Data
            text, links = await get_task_context(current_url)
            files = await download_files(links)

            # ... Step B: Ask AI
            plan = await generate_solution(text, files)
            code = plan.get("python_code")
            submit_url = plan.get("submit_url")

            # --- DEBUG CODE ---
            print("\n--- AI GENERATED CODE START ---")
            print(code)
            print("--- AI GENERATED CODE END ---\n")
            # ------------------

            # CRITICAL: Step B/C - Robust URL Extraction and Validation
            url_found = False

            # 1. Check AI result first (prefers AI's output)
            if submit_url and submit_url.startswith('http'):
                url_found = True
            else:
                # 2. Fallback: Search the entire page text for either a full URL or a relative path
                print("⚠️ AI missed the submit URL, applying dynamic fallback search...")
                
                # Regex 1: Look for any relative path that starts with a slash (/) and contains 'submit', 'answer', or 'post'
                # This is more dynamic than just looking for '/submit'
                match_relative = re.search(r'(/[^/\s]+(?:submit|answer|post|quiz)[^\s]*)', text, re.IGNORECASE)
                
                # Regex 2: Look for full https:// or http:// URL
                match_absolute = re.search(r'(https?://[^\s]+)', text, re.IGNORECASE)

                if match_relative:
                    # **NEW DYNAMIC LOGIC:** Use urljoin to correctly combine base URL and relative path
                    relative_path = match_relative.group(1).strip().rstrip('.,\'"')
                    
                    # urljoin handles combining the current page's base with the relative path gracefully.
                    submit_url = urljoin(current_url, relative_path)
                    url_found = True
                    print(f"✅ Resolved relative URL dynamically to: {submit_url}")
                    
                elif match_absolute:
                    # Standard absolute URL logic (cleanup is still required)
                    submit_url = match_absolute.group(1).strip().rstrip('.,\'"')
                    
                    # Final check to ensure the URL looks like a submission endpoint
                    if any(keyword in submit_url.lower() for keyword in ["submit", "answer", "demo"]):
                        url_found = True
                    else:
                        submit_url = None # Not a submission URL, likely a file link
                        
            # 3. IF URL IS STILL MISSING: STOP THE LOOP
            if not url_found:
                print("❌ CRITICAL FAILURE: Could not find a submission URL after AI and dynamic regex search. Stopping agent.")
                # Log the text so you can inspect where the URL is
                print("--- PAGE TEXT SNIPPET (First 1000 characters) ---")
                print(text[:1000]) 
                print("--------------------------------------------------")
                break # Stop the while loop
                
            # Step C: Execute (using the updated function that returns "CODE_EXECUTION_FAILED")
            raw_answer = execute_code(code)
            
            # Clean answer (remove quotes if AI printed them)
            if raw_answer:
                answer = raw_answer.replace('"', '').strip()
                # Try converting to number if applicable
                if answer.isdigit(): answer = int(answer)
            else:
                answer = "Error"

            print(f"💡 Answer: {answer}")

            # Step D: Submit
            payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
            result = await submit_answer(submit_url, payload)
            
            # Step E: Next?
            if result.get("correct"):
                print(result.get("correct"))
                print("✅ Correct!")
                current_url = result.get("url")
            else:
                print(f"❌ Wrong: {result.get('reason')}")
                current_url = result.get("url") # Skip to next even if wrong (per prompt rules)

            # Cleanup
            for f in files: os.remove(f)

        except Exception as e:
            print(f"🔥 Error: {e}")
            traceback.print_exc()
            break