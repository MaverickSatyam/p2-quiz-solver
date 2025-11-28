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
AI_MODEL = "openai/gpt-4.1-nano"  # Use high-quality model for coding tasks

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
async def generate_solution(instruction, file_paths, current_url_base=None):
    files_context = "\n".join([f"- {path}" for path in file_paths])
    base_url_context = f"\nCURRENT PAGE BASE URL: {current_url_base}" if current_url_base else ""
    
    system_prompt = f"""
        You are an autonomous Python data analyst. Your ONLY goal is to solve the task.
        
        You MUST respond with a single JSON object. DO NOT include any text, conversation, or markdown (```json) outside of the final JSON object.
        
        The JSON object MUST have two keys: "submit_url" (string) and "python_code" (string).
        
        Rules for the 'python_code' value:
        1. The code must be a single, complete Python script string (do NOT wrap it in ```python).
        2. The script must read the instructions and files, calculate the final answer.
        3. The script MUST print the FINAL ANSWER to stdout ONLY.
        
        Rules for the 'submit_url' value:
        1. Extract the submission URL from the instructions. It can be relative (e.g., /submit) or absolute.
        2. If no files are listed in the Context Files section, ignore the file_paths.
        
        Context Files:
        {files_context}
        """
    
    user_prompt = f"""
    Context Files:
    {files_context}
    {base_url_context}
    
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

# --- 4b. AI ANSWER DELEGATOR ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), 
       retry=retry_if_exception_type(json.JSONDecodeError))
async def delegate_to_llm_answer(instruction, file_paths, current_url_base=None):
    """Asks the AI to solve the task directly and output ONLY the answer."""
    files_context = "\n".join([f"- {path}" for path in file_paths])
    base_url_context = f"\nCURRENT PAGE BASE URL: {current_url_base}" if current_url_base else ""
    
    # System prompt to force a direct answer
    system_prompt = f"""
        You are an expert data analyst. Your ONLY goal is to solve the task described in the Instruction using the Context.
        You MUST provide the final answer as a clean string. DO NOT include any text, conversation, code blocks, or markdown.
        
        Context Files:
        {files_context}
        """
    
    user_prompt = f"""
    {base_url_context}
    
    Instruction:
    {instruction}
    
    Provide the final numerical or string answer ONLY:
    """

    response = await client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Clean up the raw response text
    raw_answer = response.choices[0].message.content.strip()
    return raw_answer.replace('"', '').strip()

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

# --- Helper function for answer normalization ---
def normalize_answer(raw_answer):
    """Cleans up and converts the answer to the appropriate format (float/int/string)."""
    if not raw_answer or raw_answer == "CODE_EXECUTION_FAILED":
        return None
        
    final_answer = raw_answer.replace('"', '').strip()
    try:
        # Check if it is a number (integer or float)
        if final_answer.isdigit() or final_answer.count('.') == 1 and final_answer.replace('.', '').isdigit():
            return float(final_answer)
        else:
            return final_answer
    except:
        return raw_answer # Fallback to original raw string


# --- 7. MAIN LOOP ---
async def run_quiz_agent(start_url, email, secret):
    current_url = start_url
    
    while current_url:
        print(f"\n🔵 Processing: {current_url}")
        
        # We will attempt Phase 1 first.
        result = {"correct": False} # Initialize result to trigger logic
        
        try:
            # --- PHASE 0: Initial Setup ---
            text, links = await get_task_context(current_url)
            files = await download_files(links)
            url_parts = urlparse(current_url)
            base_url = f"{url_parts.scheme}://{url_parts.netloc}"
            
            # We need the submit URL for both phases. Generate the plan once for the URL.
            plan_for_url = await generate_solution(text, files, base_url)
            current_submit_url = plan_for_url.get("submit_url")
            
            # --- URL VALIDATION (REQUIRED BEFORE ANY SUBMISSION) ---
            url_found = False
            submit_url = current_submit_url
            
            
            # 1. Check AI result first (prefers AI's output)
            if submit_url and submit_url.startswith('https'):
                url_found = True
            else:
                # 2. Fallback: Search the entire page text for either a full URL or a relative path
                print("⚠️ AI missed the submit URL, applying dynamic fallback search...")
                match_relative = re.search(r'(/submit[^\s]*)', text, re.IGNORECASE)
                match_absolute = re.search(r'(https?://[^\s]+)', text, re.IGNORECASE)

                if match_relative:
                    relative_path = match_relative.group(1).strip().rstrip('.,\'"')
                    submit_url = urljoin(base_url, relative_path)
                    url_found = True
                    print(f"✅ Resolved relative URL dynamically to: {submit_url}")
                elif match_absolute:
                    submit_url = match_absolute.group(1).strip().rstrip('.,\'"')
                    if any(keyword in submit_url.lower() for keyword in ["submit", "answer", "demo"]):
                        url_found = True
                    else:
                        submit_url = None
                        
            # 3. IF URL IS STILL MISSING: BREAK
            if not url_found or not submit_url:
                print("❌ CRITICAL FAILURE: Could not find a submission URL after AI and dynamic regex search. Stopping agent.")
                break 

            # --- ATTEMPT 1: LLM DELEGATION (Primary Method) ---
            answer_source = "Delegation"
            try:
                print(f"\n✨ Attempt 1 ({answer_source}): LLM Delegation (Direct Answer)")
                raw_answer = await delegate_to_llm_answer(text, files, base_url)
                answer = normalize_answer(raw_answer)
                
                if answer:
                    print(f"💡 Answer ({answer_source}): {answer}")
                    payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                    result = await submit_answer(submit_url, payload)
                    
                    if result.get("correct"):
                        print("✅ Correct! (Delegation)")
                        current_url = result.get("url")
                        continue # Move to next loop iteration
                    else:
                        print(f"❌ Wrong ({answer_source}): {result.get('reason')}")
                        # If wrong, proceed to Attempt 2.
                else:
                    print("⚠️ Delegation failed to produce a valid answer. Proceeding to Attempt 2.")

            except Exception as e:
                print(f"⚠️ Delegation attempt failed: {e}. Proceeding to Attempt 2.")
            
            
            # --- ATTEMPT 2: AI CODE EXECUTION (Fallback Method) ---
            if not result.get("correct"):
                answer_source = "Code"
                print(f"\n💻 Attempt 2 ({answer_source}): Falling back to AI Code Generation and Execution")
                
                # Use the plan already generated, or regenerate the code part if the URL plan was old
                plan = await generate_solution(text, files, base_url) 
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
            # Whether Attempt 2 was successful or failed, we must proceed to the next URL.
            current_url = result.get("url") 
            
            # Cleanup
            for f in files: os.remove(f)

        except Exception as e:
            print(f"🔥 Critical Error during processing: {e}")
            traceback.print_exc()
            break