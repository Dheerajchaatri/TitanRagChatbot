import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- FIXED IMPORTS ---
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="My Persona AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek modern UI
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ My Digital Twin Chatbot")
st.caption("A RAG-powered assistant trained to talk, think, and respond just like me.")

# --- SIDEBAR & API KEY ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.markdown("---")
    st.subheader("📝 Persona Rules")
    persona_instructions = st.text_area(
        "Define your persona / speech habits:",
        value=(
            "1. Speak casually with confidence and a touch of wit.\n"
            "2. Keep responses concise and directly address the query.\n"
            "3. Use slang or phraseology natural to me.\n"
            "4. If unsure about facts, answer grounded in the retrieved context."
        ),
        height=150
    )

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to get started.", icon="🗝️")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# --- RAG INITIALIZATION & VECTOR STORE ---
@st.cache_resource(show_spinner="Indexing knowledge base...")
def initialize_vector_store():
    data_path = "./data"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        # Create default mock data if folder is empty
        os.makedirs(data_path, exist_ok=True)
        with open(f"{data_path}/about_me.txt", "w", encoding="utf-8") as f:
            f.write("I am an innovative developer passionate about AI, automation, and modern web apps.")

    loader = DirectoryLoader(data_path, glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = initialize_vector_store()

# --- LLM & PROMPT PIPELINE ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Context contextualization prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# System Prompt embodying persona + RAG
system_prompt = (
    "You are a digital twin acting as me. Respond strictly adopting my tone, persona, and style.\n"
    "Follow these strict instructions:\n"
    "{persona_instructions}\n\n"
    "Use the following pieces of retrieved context to inform your answer:\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- CHAT HISTORY & SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display prior chat messages
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- USER INPUT & RESPONSE ---
if user_query := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history,
                "persona_instructions": persona_instructions
            })
            
            answer = response["answer"]
            st.markdown(answer)

    # Update persistent memory
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))
