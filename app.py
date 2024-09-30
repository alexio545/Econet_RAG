import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware

# Import the RAG functions
from rag import rag, submit_feedback
from db import get_db_connection

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key"))

# Pydantic models for request bodies
class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    feedback: int

# API key header for authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@app.on_event("startup")
async def startup_event():
    # Check database connection on startup
    conn = get_db_connection()
    if conn is None:
        logging.error("Failed to connect to the database. Please check your database configuration.")
    else:
        conn.close()
        logging.info("Successfully connected to the database.")

def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key is None or api_key == os.getenv("API_KEY", "your-api-key"):
        return api_key
    raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/ask")
async def ask_question(request: Request, question_request: QuestionRequest, api_key: Optional[str] = Depends(get_api_key)):
    try:
        logging.info(f"Received question: {question_request.question}")
        
        # Generate answer using RAG
        answer_data = rag(question_request.question)
        
        logging.info(f"Generated answer for conversation ID: {answer_data['id']}")
        
        # Store conversation ID in session
        request.session["conversation_id"] = answer_data['id']
        
        return {
            "conversation_id": answer_data['id'],
            "answer": answer_data['answer'],
            "relevance": answer_data['relevance'],
            "response_time": answer_data['response_time'],
            "total_tokens": answer_data['total_tokens'],
            "estimated_cost": answer_data['openai_cost']
        }
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your question")

@app.post("/feedback")
async def process_feedback(request: Request, feedback_request: FeedbackRequest, api_key: Optional[str] = Depends(get_api_key)):
    try:
        # Retrieve conversation ID from session
        conversation_id = request.session.get("conversation_id")
        if not conversation_id:
            raise HTTPException(status_code=400, detail="No active conversation found")
        
        logging.info(f"Received feedback for conversation ID: {conversation_id}")
        
        if feedback_request.feedback not in [-1, 1]:
            raise HTTPException(status_code=400, detail="Invalid feedback value. Must be -1 or 1")
        
        # Submit feedback to the database
        result = submit_feedback(conversation_id, feedback_request.feedback)
        
        if result['status'] == 'success':
            return {"message": "Feedback received and saved successfully"}
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your feedback")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)