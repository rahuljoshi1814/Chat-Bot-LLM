import sys
import os
import gradio
from milvus import default_server
from pymilvus import connections, Collection
import utils.model_llm_utils as model_llm
import utils.vector_db_utils as vector_db
import utils.model_embedding_utils as model_embedding

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Main function to configure and launch the Gradio QA app.
    """
    print("Configuring Gradio app...")
    
    demo = gradio.Interface(
        fn=get_responses, 
        inputs=gradio.Textbox(label="Question", placeholder="Ask a question here..."),
        outputs=[
            gradio.Textbox(label="Asking LLM with No Context"),
            gradio.Textbox(label="Asking LLM with Context (RAG)")
        ],
        examples=[
            "What are the services available at Changi Airport?",
            "How do I find Jewel at Changi?",
            "What is the baggage allowance for Changi flights?",
            "What shops are there at Jewel?"
        ],
        allow_flagging="never"
    )

    print("Launching Gradio app...")
    demo.queue().launch(
        share=False,
        show_error=True,
        server_name='127.0.0.1',
        server_port=int(os.getenv('CDSW_APP_PORT', 7860))  # Default to port 7860 if CDSW_APP_PORT is not set
    )
    print("Gradio app is ready.")

def get_responses(question):
    """
    Generates responses for a given question using the LLM with and without context.

    Parameters:
    - question (str): The user's question.

    Returns:
    - Tuple of responses: (plain_response, rag_response)
    """
    try:
        # Load Milvus Vector DB collection
        vector_db_collection = Collection('changi_jewel_docs')
        vector_db_collection.load()

        # Retrieve nearest knowledge base chunk
        context_chunk = get_nearest_chunk_from_vectordb(vector_db_collection, question)
        vector_db_collection.release()

        # Create enhanced prompts
        prompt_with_context = create_enhanced_prompt(context_chunk, question)
        prompt_without_context = create_enhanced_prompt("none", question)

        # Generate responses from LLM
        rag_response = get_llm_response(prompt_with_context)
        plain_response = get_llm_response(prompt_without_context)

        return plain_response, rag_response
    except Exception as e:
        print(f"Error generating responses: {e}")
        return "Error: Unable to generate response.", "Error: Unable to generate response."

def get_nearest_chunk_from_vectordb(vector_db_collection, question):
    """
    Retrieves the nearest knowledge base chunk for the given question from the vector database.

    Parameters:
    - vector_db_collection: Milvus collection object.
    - question (str): The user's question.

    Returns:
    - str: The text of the nearest knowledge base chunk.
    """
    try:
        # Generate embedding for the question
        question_embedding = model_embedding.get_embeddings(question)

        # Define search parameters
        vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        # Search the vector database
        nearest_vectors = vector_db_collection.search(
            data=[question_embedding],
            anns_field="embedding",
            param=vector_db_search_params,
            limit=1,
            expr=None,
            output_fields=['relativefilepath'],
            consistency_level="Strong"
        )

        # Get the file path of the nearest chunk
        id_path = nearest_vectors[0].ids[0]
        print(f"Nearest vector file path: {id_path}")

        # Load the knowledge base chunk
        return load_context_chunk_from_data(id_path)
    except Exception as e:
        print(f"Error retrieving nearest chunk: {e}")
        return "none"

def load_context_chunk_from_data(id_path):
    """
    Loads the knowledge base chunk text from the given file path.

    Parameters:
    - id_path (str): The file path to the knowledge base chunk.

    Returns:
    - str: The text content of the chunk.
    """
    try:
        if not os.path.exists(id_path):
            raise FileNotFoundError(f"File not found: {id_path}")
        with open(id_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading context chunk: {e}")
        return ""

def create_enhanced_prompt(context, question):
    """
    Creates an enhanced prompt for the LLM.

    Parameters:
    - context (str): The context retrieved from the knowledge base.
    - question (str): The user's question.

    Returns:
    - str: The enhanced prompt.
    """
    prompt_template = "<human>: %s. Answer this question based on given context: %s\n<bot>:"
    return prompt_template % (context, question)

def get_llm_response(prompt):
    """
    Gets a response from the LLM for the given prompt.

    Parameters:
    - prompt (str): The prompt to send to the LLM.

    Returns:
    - str: The LLM's response.
    """
    try:
        stop_words = ['<human>:', '\n<bot>:']
        generated_text = model_llm.get_llm_generation(
            prompt,
            stop_words=stop_words,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.85,
            top_k=70,
            repetition_penalty=1.07
        )
        return generated_text
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "Error: Unable to generate LLM response."

if __name__ == "__main__":
    main()

