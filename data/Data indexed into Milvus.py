from pymilvus import Collection, connections
import utils.model_embedding_utils as model_embedding # Assuming this is your embedding generation module
import os
from utils import model_embedding_utils as model_embedding
# Connect to Milvus
connections.connect()

# Index the scraped data into Milvus
def index_data_to_milvus(data_path, collection_name):
    # Create a collection if it doesn't exist
    if collection_name not in [coll.name for coll in Collection.list()]:
        collection = Collection(collection_name)
    else:
        collection = Collection(collection_name)

    # Iterate over files in the folder and index them
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    document = file.read()
                    embedding = model_embedding.get_embeddings(document)
                    # Insert embeddings into Milvus (assumes you have defined fields)
                    collection.insert([embedding])

# Index new data
index_data_to_milvus('changi_jewel_docs', 'changi_jewel_docs')

print("Data indexed into Milvus")
