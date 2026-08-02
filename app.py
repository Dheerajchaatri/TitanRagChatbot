import os
import shutil
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)


# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="Titan Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ===================================================
# CUSTOM CSS
# ===================================================

st.markdown(
    """
<style>

.stApp{
    background:#0e1117;
    color:white;
}

h1,h2,h3,h4,h5,h6,p,label,span{
    color:white !important;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stChatMessage"]{
    background:#1e293b;
    border-radius:15px;
    padding:14px;
    margin-bottom:10px;
}

[data-testid="stChatMessage"] p{
    color:white !important;
}

[data-testid="stChatInput"]{
    background:#111827 !important;
    border:1px solid #334155;
    border-radius:14px;
}

[data-testid="stChatInput"] textarea{
    color:white !important;
    background:#111827 !important;
    -webkit-text-fill-color:white !important;
    caret-color:white !important;
}

[data-testid="stChatInput"] textarea::placeholder{
    color:#94a3b8 !important;
}

button{
    border-radius:10px !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ===================================================
# TITLE
# ===================================================

st.title("⚡ My Digital Twin Chatbot")

st.caption(
    "A RAG-powered chatbot that can answer from TXT files and uploaded PDFs."
)


# ===================================================
# API KEY
# ===================================================

try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = None


with st.sidebar:

    st.header("⚙️ Configuration")

    if api_key:
        st.success("Groq API Loaded ✅")
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
        )

    st.divider()

    st.subheader("📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
    )

    if uploaded_pdf is not None:

        upload_folder = "./data/uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True,
        )

        pdf_path = os.path.join(
            upload_folder,
            uploaded_pdf.name,
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # delete previous database
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

        st.success("PDF Uploaded Successfully ✅")

        st.cache_resource.clear()

        st.rerun()

    st.divider()

    persona_instructions = st.text_area(
        "Persona Rules",
        value="""
1. Speak casually.
2. Keep answers concise.
3. Use retrieved context.
4. If answer isn't available, say you don't know.
""",
        height=170,
    )


if not api_key:

    st.warning("Please enter your Groq API Key.")

    st.stop()


# ===================================================
# VECTOR DATABASE
# ===================================================

@st.cache_resource(show_spinner=False)
def initialize_vector_store():

    data_folder = "./data"
    upload_folder = "./data/uploads"
    db_folder = "./chroma_db"

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(upload_folder, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Load Existing Database
    # -----------------------------

    if os.path.exists(db_folder) and len(os.listdir(db_folder)) > 0:

        vectorstore = Chroma(
            persist_directory=db_folder,
            embedding_function=embeddings,
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    # -----------------------------
    # Build New Database
    # -----------------------------

    documents = []

    # TXT FILES

    for file in os.listdir(data_folder):

        path = os.path.join(data_folder, file)

        if file.endswith(".txt"):

            loader = TextLoader(
                path,
                encoding="utf-8",
            )

            documents.extend(loader.load())

    # PDF FILES

    for file in os.listdir(upload_folder):

        path = os.path.join(upload_folder, file)

        if file.endswith(".pdf"):

            loader = PyPDFLoader(path)

            documents.extend(loader.load())

    if len(documents) == 0:

        st.error("No TXT or PDF found inside data folder.")

        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_folder,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


retriever = initialize_vector_store()


# ===================================================
# GROQ MODEL
# ===================================================

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.4,
)

# ===================================================
# RAG PIPELINE
# ===================================================

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and latest user question, rewrite it into a standalone question. Do not answer it."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


history_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)


system_prompt = """
You are Titan AI.

Answer ONLY from the retrieved context.

If the answer is not present in the provided context,
reply:

"I couldn't find this information in the uploaded documents."

Personality Rules:

{persona}

Retrieved Context:

{context}
"""


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(
    persona=persona_instructions
)


question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)


rag_chain = create_retrieval_chain(
    history_retriever,
    question_answer_chain,
)


# ===================================================
# CHAT MEMORY
# ===================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ===================================================
# SHOW OLD CHAT
# ===================================================

for message in st.session_state.chat_history:

    role = (
        "user"
        if isinstance(message, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):
        st.markdown(message.content)


# ===================================================
# CHAT INPUT
# ===================================================

user_prompt = st.chat_input(
    "Ask anything from your uploaded PDF..."
)

if user_prompt:

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = rag_chain.invoke(
                    {
                        "input": user_prompt,
                        "chat_history": st.session_state.chat_history,
                    }
                )

                answer = result.get(
                    "answer",
                    "Sorry, I couldn't generate a response."
                )

            except Exception as e:

                answer = f"❌ Error: {str(e)}"

            st.markdown(answer)

    st.session_state.chat_history.append(
        HumanMessage(content=user_prompt)
    )

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

