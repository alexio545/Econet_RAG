import time
import random
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from db import save_conversation, save_feedback, get_db_connection

# Set the timezone to CET (Europe/Berlin)
tz = ZoneInfo("Europe/Berlin")

# List of sample questions and answers
SAMPLE_QUESTIONS = [
    "I purchased a data bundle but it's not yet credited to my account.",
    "Hi, I just purchased my daily bundles and they have exhausted though I haven't used them much.",
    "Can you send me data settings to my phone?",
    "I have WhatsApp bundle but am failing to make an App call?",
    "I have not used my line for about 3 months now and I am failing to make calls or send messages?",
]

SAMPLE_ANSWERS = [
    "Log your query on your self-service platform, https://selfcare.econet.co.zw under 'My Queries'. Provide your full name and mobile number.",
    "We would like to inform you that all our bundles are usage-based, and you can now track your data, airtime, or SMS usage via My Web self-care. You just need to follow this link: https://selfcare.econet.co.zw/ and register.",
    "Send an SMS 'Device Type' followed by 'Settings' (e.g., 'Android Settings') to 222 to receive manual settings for your phone.",
    "Please note that you cannot make calls using the WhatsApp bundle. In order to make calls, you will need data bundles.",
]

MODELS = ["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "gemma2-9b-it", "llama3-70b-8192"]
RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]

def generate_synthetic_data(start_time, end_time):
    current_time = start_time
    conversation_count = 0
    print(f"Starting historical data generation from {start_time} to {end_time}")
    while current_time < end_time:
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        model = random.choice(MODELS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        if model.startswith("openai/"):
            openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "model_used": model,
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, current_time)
        print(
            f"Saved conversation: ID={conversation_id}, Time={current_time}, Model={model}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        current_time += timedelta(minutes=random.randint(1, 15))
        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} conversations so far...")

    print(
        f"Historical data generation complete. Total conversations: {conversation_count}"
    )

def generate_live_data():
    conversation_count = 0
    print("Starting live data generation...")
    while True:
        current_time = datetime.now(tz)
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        model = random.choice(MODELS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        if model.startswith("openai/"):
            openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "model_used": model,
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, current_time)
        print(
            f"Saved live conversation: ID={conversation_id}, Time={current_time}, Model={model}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for live conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} live conversations so far...")

        time.sleep(1)

if __name__ == "__main__":
    print(f"Script started at {datetime.now(tz)}")
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(hours=6)
    print(f"Generating historical data from {start_time} to {end_time}")
    generate_synthetic_data(start_time, end_time)
    print("Historical data generation complete.")

    print("Starting live data generation... Press Ctrl+C to stop.")
    try:
        generate_live_data()
    except KeyboardInterrupt:
        print(f"Live data generation stopped at {datetime.now(tz)}.")
    finally:
        print(f"Script ended at {datetime.now(tz)}")