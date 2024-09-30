import os
from dotenv import load_dotenv

# Set environment variable to disable timezone check
os.environ['RUN_TIMEZONE_CHECK'] = '0'

# Import the init_db function from the db module
from db import init_db

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete.")