import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from solver import run_quiz_agent  # We will build this robust solver next

app = FastAPI()

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/solver")
async def entry_point(payload: QuizRequest, background_tasks: BackgroundTasks):
    # 1. Verify Secret
    if payload.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    
    # 2. Trigger the "Indefinite Loop" Agent
    # We pass the payload to the heavy lifter
    background_tasks.add_task(run_quiz_agent, payload.url, payload.email, payload.secret)

    # 3. Respond instantly (Requirement: < 1s typically)
    return {"message": "Agent started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)