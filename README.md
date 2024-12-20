# LLM-RAG Chatbot Augmented with Enterprise Data
## Overview
The LLM-RAG Chatbot is an AI-driven question-answering application that leverages Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses. This chatbot integrates a pre-trained Large Language Model (LLM) with an enterprise knowledge base stored in a vector database (Milvus) for enhanced performance.

## Features
- Context-Aware Responses: Combines LLM capabilities with relevant knowledge base chunks to generate accurate answers.
- Vector Search: Uses Milvus to retrieve the most relevant knowledge base chunks for a given query.
- Interactive UI: Built using Gradio for an easy-to-use question-and-answer interface.
- Customizable: Configurable for various enterprise data domains and document types.

## Requirements
- Software Dependencies
- Python 3.8+
- Milvus (Vector Database)
### Python Libraries:
- pymilvus
- transformers
- gradio
- numpy
- torch

## Installation
1. Clone the Repository:
   git clone https://github.com/rahuljoshi1814/Chat-Bot-LLM.git
   cd LLM_Chatbot_Augmented_with_Enterprise_Data-main
2. Set Up Virtual Environment:
   python -m venv venv
   source venv/bin/activate      # On Linux/MacOS
   venv\Scripts\activate        # On Windows
3. Install Dependencies:
   pip install -r requirements.txt
4. Start Milvus Server:
   Ensure the Milvus server is running: milvus start
5. Prepare Knowledge Base:
   run "Data indexed into Milvus.py" in the data directory. so after running this milvus_data directory will be created
   run "scrape_data.py" in the data directory
after that a folder changi_jewel_docs will be created
Use the utils.model_embedding_utils script to generate embeddings and store them in Milvus.

## Usage 
1. Run the Application:
   python app.py
2. Access the Gradio Interface:
   Open the provided local URL (e.g., http://127.0.0.1:7860/) in your web browser.
3. Ask Questions:
   Enter a question in the text box.
   View responses from the LLM with and without context.
