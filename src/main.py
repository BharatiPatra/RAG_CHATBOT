from pinecone import Pinecone
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time


st.set_page_config(page_title="🧠 Pinecone RAG UI", layout="wide")
st.markdown("""
    <style>
    .chatbox {
        background-color: #f0f2f6;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .chat-input {
        position: fixed;
        bottom: 2rem;
        width: 100%;
        max-width: 720px;
    }
    .ai-message {
        background-color: #ebf3ff;
        border-left: 4px solid #2a6edc;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .human-message {
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)
# Load environment variables (like your API keys)
load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=os.getenv(PINECONE_INDEX_NAME))


# --- App Configuration ---


# --- Model and Vector Store Initialization ---

# Initialize the LLM for generating responses
llm = ChatOpenAI(model_name="gpt-3.5-turbo") # Or use "gpt-4" if you have access

# Initialize the embeddings model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Initialize the Pinecone vector store
# This assumes you have already loaded your data into the "your_index" Pinecone index
try:
    vector_db = PineconeVectorStore(index=pc.Index(PINECONE_INDEX_NAME), embedding=embedding_model)

    # vector_db = PineconeVectorStore(index_name=PINECONE_INDEX_NAME , embedding=embedding_model)
    st.info("✅ Connected to Pinecone index.")
except Exception as e:
    st.error(f"Failed to connect to Pinecone: {e}")
    st.stop() # Stop the app if connection fails


# --- RAG Chain Definition ---

# Define the prompt template for the LLM
# prompt_template = """
# You are a helpful assistant you trained on chai docs web application which is made by hitesh choudhary. 
# Answer  the question based only on the following context:show all the header and paragraph beautiflly for user and if any  code is present in the context like .cpp or .py then beautifully format it, then return the code as it is without any explanation.
# {context}

# Question: {question}
# """
prompt_template = """
You are a smart and helpful assistant trained specifically on the Chai Docs web application made by Hitesh Choudhary. 
Based on the context provided below, answer the user's question in a **detailed, step-by-step, and beginner-friendly** manner.

🟡 **Very Important Instructions**:
- Only use the information present in the `context` below. Do NOT make up any information.
- If the answer involves code (like Python or Django), include the full code blocks inside triple backticks ``` and specify the language (e.g., ```python).
- If the context contains multiple relevant sections, summarize and merge them meaningfully.
- Use Markdown formatting for:
  - Headers (`##` or `###`)
  - Bullet points
  - Code blocks
  - Line breaks between paragraphs
- Do NOT explain anything that is not found in the context.

🧾 **Context**:
{context}

❓ **User Question**:
{question}

✍️ **Your Answer**:
"""


prompt = ChatPromptTemplate.from_template(prompt_template)

# Define the output parser
output_parser = StrOutputParser()
# Create a retriever from the vector database
retriever = vector_db.as_retriever(search_kwargs={"k": 10
                                                  
                                                  }) 
print("Retriever initialized successfully.")
print("retriever:", retriever)
# Define the RAG chain using LangChain Expression Language (LCEL)
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | output_parser
# )
from langchain_core.runnables import RunnableLambda

def retrieve_with_print(query):
    docs = retriever.invoke(query)
    print("Retrieved Docs:", [doc.page_content for doc in docs])  # Print in terminal
    return docs

rag_chain = (
    {
        "context": RunnableLambda(retrieve_with_print),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | output_parser
)



# --- Chat History Management ---

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I can help you with questions about the chai docs. What would you like to know?")
    ]

# --- User Interaction ---
st.title("📄 Chai Docs Chatbot")
st.markdown("Search what you want to know about chai docs using the power of Pinecone and LangChain!")


# Get user input from the text input box
user_query = st.text_input("🔍 Enter your question:", placeholder="e.g., What is Django?")

if user_query:
    # Add user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Get the AI response using the RAG chain
    with st.spinner("Thinking..."):
        response_text = rag_chain.invoke(user_query)

    # Add AI's response to chat history
    st.session_state.chat_history.append(AIMessage(content=response_text))
    

# --- Display Chat History ---

# Display the chat history
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)
        
