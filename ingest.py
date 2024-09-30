import json
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(file_path):
    """
    Load documents from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the documents.

    Returns:
        list: A list of documents loaded from the file.
    """
    logging.info(f"Loading documents from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ground_truth(file_path):
    """
    Load ground truth data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the ground truth data.

    Returns:
        list: A list of dictionaries representing the ground truth records.
    """
    logging.info(f"Loading ground truth from {file_path}")
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')

def initialize_model(model_name):
    """
    Initialize the SentenceTransformer model.

    Args:
        model_name (str): The name of the SentenceTransformer model to load.
    
    Returns:
        SentenceTransformer: An instance of the loaded model.
    """
    logging.info(f"Initializing model: {model_name}")
    return SentenceTransformer(model_name)

def initialize_elasticsearch(host):
    """
    Initialize the Elasticsearch client.

    Args:
        host (str): Elasticsearch host URL.

    Returns:
        Elasticsearch: An instance of the Elasticsearch client.
    """
    logging.info(f"Connecting to Elasticsearch at {host}")
    return Elasticsearch(host)

def create_index(es_client, index_name):
    """
    Create an Elasticsearch index with specific settings and mappings.

    Args:
        es_client (Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the Elasticsearch index to create.
    """
    logging.info(f"Creating index: {index_name}")
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "id": {"type": "text"},
                "question_answer_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

def ingest_documents(es_client, index_name, documents, model):
    """
    Ingest documents into the Elasticsearch index.

    Args:
        es_client (Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the Elasticsearch index.
        documents (list): List of documents to ingest.
        model (SentenceTransformer): SentenceTransformer model for embedding.
    """
    logging.info(f"Ingesting documents into index: {index_name}")
    for doc in tqdm(documents, desc="Ingesting documents"):
        question = doc['Question']
        answer = doc['Answer']
        
        es_doc = {
            "question": question,
            "answer": answer,
            "question_answer_vector": model.encode(question + ' ' + answer)
        }
        
        es_client.index(index=index_name, document=es_doc)
    logging.info("Document ingestion completed.")

def main():
    """
    Main function to load data, initialize models, create index, and ingest documents.
    """
    logging.info("Starting the data ingestion process.")

    # Load environment variables
    documents_path = os.getenv('DOCUMENTS_PATH')
    ground_truth_path = os.getenv('GROUND_TRUTH_PATH')
    model_name = os.getenv('MODEL_NAME')
    elasticsearch_host = os.getenv('ELASTIC_URL_LOCAL')
    index_name = os.getenv('INDEX_NAME')

    # Load documents and ground truth
    documents = load_documents(documents_path)
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Initialize model and Elasticsearch client
    model = initialize_model(model_name)
    es_client = initialize_elasticsearch(elasticsearch_host)
    
    # Create index and ingest documents
    create_index(es_client, index_name)
    ingest_documents(es_client, index_name, documents, model)
    
    logging.info("Data ingestion complete.")

if __name__ == "__main__":
    main()
