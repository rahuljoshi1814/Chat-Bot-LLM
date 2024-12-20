from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the LLM model you want to use
model_name = 'gpt2'  # Or any other LLM model like 'EleutherAI/gpt-neo-2.7B'

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model locally
tokenizer.save_pretrained('./models/llm-model')
model.save_pretrained('./models/llm-model')
