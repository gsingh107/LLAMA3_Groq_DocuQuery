# PDF Retrieval Chatbot
Tryt this out at:
https://docusagechatbot.streamlit.app/

## Overview

The PDF Retrieval Chatbot is a powerful application designed to process multiple PDF documents and provide intelligent responses based on the content within those documents.
Utilizing advanced language models like **Llama 3 and integrating with GroqAPI**, this chatbot employs **Langchain** for efficient document retrieval techniques. 
Users can interactively ask questions and receive relevant information extracted from uploaded PDFs, making it an essential tool for knowledge discovery.

![image](https://github.com/user-attachments/assets/596a8b57-756b-4a6f-83b1-1a1f3629e7df)


## Features
### Multi-PDF Support: 
Upload multiple PDF files at once and let the chatbot process them simultaneously.
### Natural Language Queries: 
Ask questions in natural language and get precise answers from the documents.
### Contextual Understanding: 
The chatbot retrieves information based on the context of the query, providing relevant responses.
### User-Friendly Interface: 
Built with Streamlit for an intuitive and interactive user experience. 

## Installation



### Clone the Repository

git clone https://github.com/gsingh107/LLAMA3_Groq_DocuQuery

Generat the Groq API key from https://console.groq.com/keys for Llama3-70b

### Install Dependencies

pip install -r requirements.txt

## Usage
### 1. Run the Application:

streamlit run app.py

### 2. Upload PDFs: Use the sidebar to upload your PDF documents.
### 3. Ask Questions: Type your queries in the input box and receive answers based on the PDF content.
### 4. Example Queries
"What are the main topics covered in these documents?"
"Can you summarize the second PDF?"
"What does the author say about [specific topic]?"
