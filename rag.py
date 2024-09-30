import os
import time
import uuid
from typing import Dict, Any, Tuple
import logging
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from groq import Groq
from db import save_conversation, save_feedback

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize model and Elasticsearch client
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
es_client = Elasticsearch('http://localhost:9200')

def elastic_search_knn(field, vector, index_name="customer-support"):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000
    }
    search_query = {
        "knn": knn,
        "_source": ["question", "answer"]
    }
    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    return [hit['_source'] for hit in es_results['hits']['hits']]

def question_answer_vector_knn(question):
    v_q = model.encode(question)
    return elastic_search_knn('question_answer_vector', v_q)

def build_prompt(query, search_results):
    prompt_template = """
    You are a highly knowledgeable, friendly, and empathetic and helpful customer support assistant for a telecommunications company. Your role is to assist customers by answering their questions with accurate, clear, and concise information based on the CONTEXT provided.

    Please respond in a conversational tone that makes the customer feel heard and understood and Be concise, professional, and empathetic in your responses.

    Use only the facts from the CONTEXT when answering the customer's QUESTION. If the CONTEXT does not have the context to answer the QUESTION, gently suggest that the customer reach out to a live support agent for further assistance.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()
    
    context = "\n\n".join([f"question: {doc['question']}\nanswer: {doc['answer']}" for doc in search_results])
    
    return prompt_template.format(question=query, context=context).strip()

def llm(prompt, model='llama-3.1-70b-versatile'):
    client = Groq()
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.time()
    
    return response.choices[0].message.content, response.usage.to_dict(), end_time - start_time

def calculate_openai_cost(tokens: int, model: str) -> float:
    # This is a placeholder function. Implement the actual cost calculation
    # based on OpenAI's pricing for the equivalent models.
    return tokens * 0.00002  # Example cost calculation

def evaluate_relevance(question: str, answer: str) -> Tuple[str, str, Dict[str, int]]:
    logging.info("Evaluating relevance of generated answer...")
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt)

    try:
        json_eval = json.loads(evaluation)
        logging.info(f"Evaluation result: {json_eval}")
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        logging.error("Failed to parse evaluation JSON.")
        return "UNKNOWN", "Failed to parse evaluation", tokens

def rag(query, model='llama-3.1-70b-versatile') -> Dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    question = query if isinstance(query, str) else query.get('Question') or query.get('question')
    if not question:
        raise ValueError("Input must be a string or a dictionary with a 'Question' or 'question' key")
    
    search_results = question_answer_vector_knn(question)
    prompt = build_prompt(question, search_results)
    
    answer, tokens, response_time = llm(prompt, model=model)
    
    relevance, relevance_explanation, eval_tokens = evaluate_relevance(question, answer)
    
    openai_cost = calculate_openai_cost(tokens['total_tokens'] + eval_tokens['total_tokens'], model)
    
    conversation_data = {
        "id": conversation_id,
        "question": question,
        "answer": answer,
        "model_used": model,
        "response_time": response_time,
        "relevance": relevance,
        "relevance_explanation": relevance_explanation,
        "prompt_tokens": tokens['prompt_tokens'],
        "completion_tokens": tokens['completion_tokens'],
        "total_tokens": tokens['total_tokens'],
        "eval_prompt_tokens": eval_tokens['prompt_tokens'],
        "eval_completion_tokens": eval_tokens['completion_tokens'],
        "eval_total_tokens": eval_tokens['total_tokens'],
        "openai_cost": openai_cost
    }
    
    save_conversation(conversation_id, question, conversation_data)
    
    return conversation_data

def submit_feedback(conversation_id: str, feedback: int) -> Dict[str, Any]:
    try:
        save_feedback(conversation_id, feedback)
        return {"status": "success", "message": "Feedback saved successfully"}
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        return {"status": "error", "message": f"Failed to save feedback: {str(e)}"}

def get_answer_for_question(question):
    return rag(question)

if __name__ == "__main__":
    # Set up the Groq API key
    os.environ["GROQ_API_KEY"] = "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"
    
    custom_question = input("Enter your question: ")
    answer_data = get_answer_for_question(custom_question)
    print(f"\nQuestion: {custom_question}")
    print(f"Answer: {answer_data['answer']}")
    print(f"\nRelevance: {answer_data['relevance']}")
    print(f"Explanation: {answer_data['relevance_explanation']}")
    print(f"\nResponse Time: {answer_data['response_time']:.2f} seconds")
    print(f"Total Tokens: {answer_data['total_tokens']}")
    print(f"Estimated Cost: ${answer_data['openai_cost']:.6f}")
    
    feedback = input("Was this answer helpful? (1 for yes, -1 for no): ")
    feedback_result = submit_feedback(answer_data['id'], int(feedback))
    print(feedback_result['message'])