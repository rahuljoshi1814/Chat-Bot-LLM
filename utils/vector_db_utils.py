from pymilvus import connections, Collection, utility
from milvus import default_server
import os

# Start Milvus Vector DB (ensure it’s running)
default_server.stop()
default_server.set_base_dir('milvus-data')
default_server.start()

# Connect to Milvus server
try:
    connections.connect(alias='default', host='localhost', port=default_server.listen_port)
except Exception as e:
    default_server.stop()
    raise e

print("Milvus server connected successfully!")

# Ensure collection exists or create it
def create_collection(collection_name, fields):
    if collection_name not in [coll.name for coll in Collection.list()]:
        collection = Collection(name=collection_name, fields=fields)
    else:
        collection = Collection(name=collection_name)
    return collection

# Insert vectors into Milvus
def insert_vectors(collection_name, vectors, ids=None):
    collection = Collection(name=collection_name)
    collection.insert([vectors, ids])
    print(f"Inserted {len(vectors)} vectors into Milvus collection {collection_name}")

# Query Milvus for similar vectors
def query_vectors(collection_name, query_vector, top_k=5):
    collection = Collection(name=collection_name)
    search_params = {'metric_type': 'L2', 'params': {'nprobe': 10}}
    results = collection.search([query_vector], 'embedding', search_params, top_k)
    return results

# Test if Milvus is running and connected
print(f"Milvus server version: {utility.get_server_version()}")
