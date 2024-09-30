

import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import logging




from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


RUN_TIMEZONE_CHECK = os.getenv('RUN_TIMEZONE_CHECK', '1') == '1'

TZ_INFO = os.getenv("TZ", "Europe/Berlin")
tz = ZoneInfo(TZ_INFO)

def get_db_connection():
       try:
           print(f"Attempting to connect to database: host={os.getenv('POSTGRES_HOST_LOCAL')}, db={os.getenv('POSTGRES_DB')}, user={os.getenv('POSTGRES_USER')}")
           conn = psycopg2.connect(
               host=os.getenv("POSTGRES_HOST_LOCAL"),
               database=os.getenv("POSTGRES_DB"),
               user=os.getenv("POSTGRES_USER"),
               password=os.getenv("POSTGRES_PASSWORD"),
           )
           print("Successfully connected to the database")
           return conn
       except psycopg2.OperationalError as e:
           print(f"Unable to connect to the database. Error details: {e}")
           return None

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

def save_conversation(conversation_id, question, answer_data, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp
                ),
            )
        conn.commit()
    finally:
        conn.close()

def save_feedback(conversation_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()

def get_recent_conversations(limit=5, relevance=None):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            if relevance:
                query += f" WHERE c.relevance = %s"
            query += " ORDER BY c.timestamp DESC LIMIT %s"

            params = (relevance, limit) if relevance else (limit,)
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()

def get_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()

def get_relevance_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    relevance,
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
                FROM conversations
                GROUP BY relevance
            """)
            return cur.fetchall()
    finally:
        conn.close()

def get_avg_response_time():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT AVG(response_time) FROM conversations")
            return cur.fetchone()[0]
    finally:
        conn.close()

def get_model_usage_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    model_used,
                    COUNT(*) as count,
                    AVG(response_time) as avg_response_time,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(openai_cost) as total_cost
                FROM conversations
                GROUP BY model_used
            """)
            return cur.fetchall()
    finally:
        conn.close()

def get_token_usage_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(openai_cost) as total_cost
                FROM conversations
            """)
            return cur.fetchone()
    finally:
        conn.close()

def check_timezone():
    conn = get_db_connection()
    if conn is None:
        logging.error("Skipping timezone check due to database connection failure.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW timezone;")
            db_timezone = cur.fetchone()[0]
            logging.info(f"Database timezone: {db_timezone}")

            cur.execute("SELECT current_timestamp;")
            db_time_utc = cur.fetchone()[0]
            logging.info(f"Database current time (UTC): {db_time_utc}")

            db_time_local = db_time_utc.astimezone(tz)
            logging.info(f"Database current time ({TZ_INFO}): {db_time_local}")

            py_time = datetime.now(tz)
            logging.info(f"Python current time: {py_time}")

            # Use py_time instead of tz for insertion
            cur.execute("""
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING timestamp;
            """, 
            ('test', 'test question', 'test answer', 'test model', 0.0, 0.0, 
             'test explanation', 0, 0, 0, 0, 0, 0, 0.0, py_time))

            inserted_time = cur.fetchone()[0]
            logging.info(f"Inserted time (UTC): {inserted_time}")
            logging.info(f"Inserted time ({TZ_INFO}): {inserted_time.astimezone(tz)}")

            cur.execute("SELECT timestamp FROM conversations WHERE id = 'test';")
            selected_time = cur.fetchone()[0]
            logging.info(f"Selected time (UTC): {selected_time}")
            logging.info(f"Selected time ({TZ_INFO}): {selected_time.astimezone(tz)}")

            # Clean up the test entry
            cur.execute("DELETE FROM conversations WHERE id = 'test';")
            conn.commit()
    except Exception as e:
        logging.error(f"An error occurred during timezone check: {e}")
        conn.rollback()
    finally:
        conn.close()

if RUN_TIMEZONE_CHECK:
    check_timezone()
else:
    logging.info("Timezone check is disabled.")