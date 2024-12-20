from transformers import AutoTokenizer, AutoModel
import os

# Define the model you want to use for embeddings (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Define the path where you want to store the model
save_dir = 'models/embedding-model'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download and load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save the model and tokenizer locally
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Model saved to {save_dir}")
