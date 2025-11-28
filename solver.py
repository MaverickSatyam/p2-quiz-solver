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

# --- 3. DOWNLOADER AND FILE READER ---
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

def get_file_contents_for_prompt(files):
    """Reads the content of downloaded files for inclusion in the prompt."""
    contents = {}
    for filepath in files:
        if os.path.getsize(filepath) > 200000: # Limit to 200KB for prompt inclusion
            contents[filepath] = "File too large to include in prompt."
            continue
            
        try:
            with open(filepath, 'r') as f:
                # Read max 20,000 characters to prevent context overflow
                contents[filepath] = f.read(20000) 
        except Exception as e:
            contents[filepath] = f"Error reading file: {e}"
            
    return "\n\n".join([f"--- FILE: {path} (Snippet) ---\n{content}" for path, content in contents.items()])

# --- 4. AI CODE GENERATOR (Modified Signature) ---
async def generate_solution(instruction, file_paths, current_url_base, all_links, previous_error=None):
    files_context = "\n".join([f"- {path}" for path in file_paths])
    base_url_context = f"\nCURRENT PAGE BASE URL: {current_url_base}" if current_url_base else ""
    links_context = "\n".join([f"- Text: {link['text']}, URL: {link['href']}" for link in all_links])

    # Dynamic Hinting based on previous error
    failure_context = ""
    if previous_error:
        failure_context = f"\n--- PREVIOUS ATTEMPT FAILED ---\nReason: {previous_error}\nYour previous answer was rejected. Write robust code to solve it."
    
    system_prompt = f"""
        You are an autonomous Python data analyst. Your ONLY goal is to solve the task.
        
        You MUST respond with a single JSON object. DO NOT include any text, conversation, or markdown (```json) outside of the final JSON object.
        
        The JSON object MUST have two keys: "submit_url" (string) and "python_code" (string).
        
        Rules for the 'python_code' value:
        1. The code must be a single, complete Python script string. DO NOT wrap it in ```python.
        2. The script must read the instructions and files, and calculate the final answer.
        3. The script MUST print the FINAL ANSWER to stdout ONLY.
        4. If using the 'requests' library for submission, it MUST use the **json=** argument.
        5. When using Pandas, do NOT hardcode column names. Access the first column using **df.iloc[:, 0]**.
        
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

# --- 4b. AI ANSWER DELEGATOR (Modified Signature) ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), 
       retry=retry_if_exception_type(json.JSONDecodeError))
async def delegate_to_llm_answer(instruction, file_paths, current_url_base, file_contents, previous_error=None):
    """Asks the AI to solve the task directly and output ONLY the answer."""
    files_context = "\n".join([f"- {path}" for path in file_paths])
    base_url_context = f"\nCURRENT PAGE BASE URL: {current_url_base}" if current_url_base else ""
    
    # Dynamic Hinting based on previous error
    failure_context = ""
    if previous_error:
        failure_context = f"\n--- PREVIOUS ATTEMPT FAILED ---\nReason: {previous_error}\nYour previous answer was rejected. Try a different approach."

    system_prompt = f"""
        You are an expert data analyst. Your ONLY goal is to solve the task described in the Instruction and Context.
        You MUST provide the final answer as a clean string. DO NOT include any text, conversation, code blocks, or markdown.
        
        Context Files:
        {files_context}
        """
    
    user_prompt = f"""
    {base_url_context}
    
    --- PAGE CONTENT ---
    {instruction}
    
    --- FILE SNIPPET DATA ---
    {file_contents}
    
    {failure_context}
    
    Provide the final numerical or string answer ONLY:
    """

    response = await client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    raw_answer = response.choices[0].message.content.strip()
    return raw_answer.replace('"', '').strip()

# --- 5. EXECUTOR ---
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

# --- Helper function for answer normalization ---
def normalize_answer(raw_answer):
    if not raw_answer or raw_answer == "CODE_EXECUTION_FAILED":
        return None
        
    final_answer = raw_answer.replace('"', '').strip()
    try:
        if final_answer.isdigit() or final_answer.count('.') == 1 and final_answer.replace('.', '').isdigit():
            return float(final_answer)
        else:
            return final_answer
    except:
        return raw_answer

# --- 7. MAIN LOOP ---
async def run_quiz_agent(start_url, email, secret):
    current_url = start_url
    
    while current_url:
        print(f"\n🔵 Processing: {current_url}")
        
        # We will attempt Phase 1 first.
        result = {"correct": False}
        previous_error = None # Tracks the reason for failure
        
        try:
            # --- PHASE 0: Initial Setup ---
            text, links = await get_task_context(current_url)
            files = await download_files(links)
            file_contents = get_file_contents_for_prompt(files)
            
            url_parts = urlparse(current_url)
            base_url = f"{url_parts.scheme}://{url_parts.netloc}"
            
            # --- URL VALIDATION (Generate plan once to find the URL) ---
            # Call generate_solution once to get the potential submit URL and links context
            plan_for_url = await generate_solution(text, files, base_url, links)
            current_submit_url = plan_for_url.get("submit_url")
            
            url_found = False
            submit_url = current_submit_url
            
            # 1. Check AI result first
            if submit_url and submit_url.startswith('https'):
                url_found = True
            else:
                # 2. Fallback: Search the entire page text for URL
                match_relative = re.search(r'(/submit[^\s]*)', text, re.IGNORECASE)
                match_absolute = re.search(r'(https?://[^\s]+)', text, re.IGNORECASE)

                if match_relative:
                    relative_path = match_relative.group(1).strip().rstrip('.,\'"')
                    submit_url = urljoin(base_url, relative_path)
                    url_found = True
                elif match_absolute:
                    submit_url = match_absolute.group(1).strip().rstrip('.,\'"')
                    if any(keyword in submit_url.lower() for keyword in ["submit", "answer", "demo"]):
                        url_found = True
                    else:
                        submit_url = None
                        
            if not url_found or not submit_url:
                print("❌ CRITICAL FAILURE: Could not find a submission URL. Stopping agent.")
                break 

            # --- ATTEMPT 1: LLM DELEGATION (Primary Method) ---
            answer_source = "Delegation"
            raw_answer = None
            
            try:
                print(f"\n✨ Attempt 1 ({answer_source}): LLM Delegation (Direct Answer)")
                # Pass all context, including file snippets
                raw_answer = await delegate_to_llm_answer(text, files, base_url, file_contents)
                answer = normalize_answer(raw_answer)
                
                if answer:
                    print(f"💡 Answer ({answer_source}): {answer}")
                    payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                    result = await submit_answer(submit_url, payload)
                    
                    if result.get("correct"):
                        print("✅ Correct! (Delegation)")
                        current_url = result.get("url")
                        continue # Move to next question
                    else:
                        previous_error = result.get('reason')
                        print(f"❌ Wrong ({answer_source}): {previous_error}")
                        # If wrong, proceed to Attempt 2.
                else:
                    print("⚠️ Delegation failed to produce a valid answer. Proceeding to Attempt 2.")

            except Exception as e:
                print(f"⚠️ Delegation attempt failed: {e}. Proceeding to Attempt 2.")
            
            
            # --- ATTEMPT 2: AI CODE EXECUTION (Fallback Method) ---
            if not result.get("correct"):
                answer_source = "Code"
                print(f"\n💻 Attempt 2 ({answer_source}): Falling back to AI Code Generation and Execution")
                
                # Pass all context, including the reason for the previous failure
                plan = await generate_solution(text, files, base_url, links, previous_error)
                code = plan.get("python_code")
                
                # --- DEBUG CODE ---
                print("\n--- AI GENERATED CODE START ---")
                print(code)
                print("--- AI GENERATED CODE END ---\n")
                # ------------------
                
                raw_answer = execute_code(code)
                answer = normalize_answer(raw_answer)

                if answer:
                    print(f"💡 Answer ({answer_source}): {answer}")
                    payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                    result = await submit_answer(submit_url, payload)
                    
                    if result.get("correct"):
                        print("✅ Correct! (Code)")
                    else:
                        print(f"❌ Wrong ({answer_source}): {result.get('reason')}")
                else:
                    print("❌ Code execution failed or produced no answer. Moving to next URL.")

            # --- FINAL LOOP CONTROL ---
            current_url = result.get("url") 
            
            # Cleanup
            for f in files: os.remove(f)

        except Exception as e:
            print(f"🔥 Critical Error during processing: {e}")
            traceback.print_exc()
            break