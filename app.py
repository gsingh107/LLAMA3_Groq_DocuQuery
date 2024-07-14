import streamlit
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from sentence_transformers import SentenceTransformer
import time
from langchain.embeddings import HuggingFaceEmbeddings
# from HTMLtemplates import css

def get_pdf_docs(pdfs):
    text= ""
    for pdf in pdfs:
        pdf_reader =PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_texts):
    text_splitter =CharacterTextSplitter(separator='\n',chunk_size=1000,
                                         chunk_overlap=200,
                                         length_function=len)
    chunks =text_splitter.split_text(raw_texts)
    return chunks

def get_vectorstores(text_chunks):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    st.sidebar.write('Vector DB is ready')
    end_time = time.time()
    st.sidebar.write(f"Time taken to create DB: {end_time - start_time:.2f} seconds")
    return vectorstore

# Function to interact with Groq AI
def chat_groq(messages):
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content
st.set_page_config(page_title='Chat with Multiple PDFs',page_icon=':books:')
# st.write(css, unsafe_allow_html=True)

def summarize_chat_history(chat_history):
    chat_history_text = " ".join([f"{chat['role']}: {chat['content']}" for chat in chat_history])
    prompt = f"Summarize the following chat history:\n\n{chat_history_text}"
    messages = [{'role': 'system', 'content': 'You are very good at summarizing the chat between User and Assistant'}]
    messages.append({'role': 'user', 'content': prompt})
    summary = chat_groq(messages)
    return summary


def main():
    load_dotenv()

    st.markdown('<h1 class="main-header">Chat with Multiple PDFs 📚 </h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents:</p>', unsafe_allow_html=True)
    user_input = st.text_area('', key="user_input", placeholder="Type your question here..."
                                  )


    # st.session_state.user_input = st.text_input('Ask questions about your documents:')
    # initializing the session state variables

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if 'conversation' not in st.session_state:
        st.session_state.conversation =None

    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []



    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Upload your PDFs here and click on Process',accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner("Processing"):
                # get pdf text
                raw_text= get_pdf_docs(pdf_docs)
                # st.write(raw_text)
                #get the text chunks
                if raw_text is "":
                    st.write("Please upload PDFs first")
                else:
                    chunks =get_text_chunks(raw_text)
                    vectorstore = get_vectorstores(chunks)
                    st.session_state.vectorstore = vectorstore
                # create vector database


    user_question = st.session_state.user_input
    # st.write(user_question)
    if st.session_state.vectorstore is not None:
        def submit_with_pdf():

            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            context = retriever.invoke(user_question)
            prompt = f'''
                        Answer the user's question based on the latest input provided in the chat history. Ignore
                        previous inputs unless they are directly related to the latest question. Provide a generic
                        answer if the answer to the user's question is not present in the context by mentioning it
                        as general information.
    
                        Context: {context}
    
                        Chat History: {st.session_state.chat_history}
    
                        Latest Question: {user_question}
                        '''

            messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            messages.append({'role': 'user', 'content': prompt})

            try:
                ai_response = chat_groq(messages)
            except Exception as e:
                st.error(f"Error occurred during chat_groq execution: {str(e)}")
                ai_response = "An error occurred while fetching response. Please try again."

            # Display the current output prompt
            st.session_state.current_prompt = ai_response

            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

            # Clear the input field
            st.session_state.user_input = ""

    def submit_without_pdf():

        if user_question:
            prompt = f'''
            Answer the user's question based on the latest input provided in the chat history. Ignore
            previous inputs unless they are directly related to the latest
            question. 

            Chat History: {st.session_state.chat_history}

            Latest Question: {user_question}
            '''

            messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            messages.append({'role': 'user', 'content': prompt})

            try:
                ai_response = chat_groq(messages)
            except Exception as e:
                st.error(f"Error occurred during chat_groq execution: {str(e)}")
                ai_response = "An error occurred while fetching response. Please try again."

            # Display the current output prompt
            st.session_state.current_prompt = ai_response

            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

            # Clear the input field
            st.session_state.user_input = ""
    # Display the current output prompt if available



    if st.session_state.vectorstore is not None:
        st.button('Submit', on_click=submit_with_pdf)
    else:
        st.button('Submit', on_click=submit_without_pdf)

    # Display the current output prompt if available
    if st.session_state.current_prompt:
        st.write(st.session_state.current_prompt)

    # Button to generate chat summary
    if st.button('Generate Chat Summary'):
        st.session_state.chat_summary = summarize_chat_history(st.session_state.chat_history)

    # Display the chat summary if available
    if st.session_state.chat_summary:
        with st.expander("Chat Summary"):
            st.write(st.session_state.chat_summary)

    # Display the last 4 messages in an expander
    with st.expander("Recent Chat History"):
        recent_history = st.session_state.chat_history[-8:][::-1]
        reversed_history = []
        for i in range(0, len(recent_history), 2):
            if i+1 < len(recent_history):
                reversed_history.extend([recent_history[i+1], recent_history[i]])
            else:
                reversed_history.append(recent_history[i])
        for chat in reversed_history:
            st.write(f"{chat['role'].capitalize()}: {chat['content']}")


if __name__ =='__main__':
    main()
