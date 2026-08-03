import os
import shutil
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_groq import (
    ChatGroq
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Titan RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""

<style>

.stApp{
    background:#0f172a;
}

.block-container{
    max-width:900px;
    padding-top:25px;
}

h1,h2,h3,h4{
    color:white !important;
}

p,label,span{
    color:#e2e8f0 !important;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stChatMessage"]{

    background:#1e293b;

    border:1px solid #334155;

    border-radius:14px;

    padding:14px;

    margin-bottom:12px;

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

    background:#111827 !important;

    color:white !important;

    -webkit-text-fill-color:white !important;

    caret-color:white !important;

}

[data-testid="stChatInput"] textarea::placeholder{

    color:#94a3b8 !important;

}

[data-testid="stChatInputSubmitButton"] button{

    background:#2563eb !important;

    color:white !important;

}

[data-testid="stFileUploader"]{

    background:#1e293b;

    border:1px solid #334155;

    border-radius:12px;

    padding:10px;

}

</style>

""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.title("⚡ Titan RAG Chatbot")

st.caption(
    "Ask questions from TXT knowledge or uploaded PDFs."
)

# =====================================================
# API KEY
# =====================================================

try:
    secret_groq_key = st.secrets["GROQ_API_KEY"]
except:
    secret_groq_key = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    if secret_groq_key:

        st.success("Groq API Loaded 🔒")

        api_key = secret_groq_key

    else:

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_..."
        )

    st.divider()

    # =====================================================
    # PDF UPLOAD
    # =====================================================

    st.subheader("📄 Upload Knowledge PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        upload_folder = "./data/uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True
        )

        pdf_path = os.path.join(
            upload_folder,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )

        st.success("✅ PDF Uploaded Successfully")

    st.divider()

    # =====================================================
    # PERSONA
    # =====================================================

    persona_instructions = st.text_area(

        "Persona Rules",

        value="""
1. Speak casually with confidence.
2. Keep answers concise.
3. Use natural language.
4. Use retrieved knowledge while answering.
5. Never make up information.
""",

        height=170
    )


# =====================================================
# CHECK API KEY
# =====================================================

if not api_key:

    st.info(
        "Please enter your Groq API Key.",
        icon="🔑"
    )

    st.stop()

# =====================================================
# VECTOR DATABASE
# =====================================================

@st.cache_resource(show_spinner="Building Knowledge Base...")
def initialize_vector_store():

    data_path = "./data"
    upload_path = "./data/uploads"
    db_path = "./chroma_db"

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(upload_path, exist_ok=True)

    documents = []

    # ================= TXT FILES =================

    for root, dirs, files in os.walk(data_path):

        for file in files:

            if file.endswith(".txt"):

                loader = TextLoader(
                    os.path.join(root, file),
                    encoding="utf-8"
                )

                documents.extend(
                    loader.load()
                )

    # ================= PDF FILES =================

    for root, dirs, files in os.walk(upload_path):

        for file in files:

            if file.endswith(".pdf"):

                loader = PyPDFLoader(
                    os.path.join(root, file)
                )

                documents.extend(
                    loader.load()
                )

    # ================= DEFAULT FILE =================

    if len(documents) == 0:

        default_path = os.path.join(
            data_path,
            "about_me.txt"
        )

        with open(
            default_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("No knowledge available.")

        loader = TextLoader(
            default_path,
            encoding="utf-8"
        )

        documents.extend(
            loader.load()
        )

    # ================= TEXT SPLITTING =================

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )

    splits = splitter.split_documents(
        documents
    )

    # ================= EMBEDDINGS =================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ================= VECTOR STORE =================

    if os.path.exists(db_path):

        shutil.rmtree(db_path)

    os.makedirs(db_path, exist_ok=True)

    vectorstore = Chroma.from_documents(

        documents=splits,

        embedding=embeddings,

        persist_directory=db_path

    )

    return vectorstore.as_retriever(

        search_kwargs={"k":3}

    )


# =====================================================
# LOAD RETRIEVER
# =====================================================

retriever = initialize_vector_store()

# =====================================================
# GROQ MODEL
# =====================================================

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)


# =====================================================
# HISTORY AWARE RETRIEVER
# =====================================================

contextualize_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
Given the chat history and the latest user question,
rewrite the question so it can be understood independently.

Do not answer the question.

Only rewrite it if necessary.
"""
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        "{input}"
    )

])


history_retriever = create_history_aware_retriever(

    llm,

    retriever,

    contextualize_prompt

)


# =====================================================
# SYSTEM PROMPT
# =====================================================

system_prompt = """
You are Titan AI Assistant.

Rules:

{persona_instructions}

Answer ONLY from the retrieved knowledge.

If the answer does not exist in the uploaded TXT files or uploaded PDFs,
reply exactly:

"I couldn't find this information in the uploaded knowledge."

Do not make up facts.

Retrieved Knowledge:

{context}
"""


# =====================================================
# QA PROMPT
# =====================================================

qa_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        system_prompt
    ),

    MessagesPlaceholder(
        "chat_history"
    ),

    (
        "human",
        "{input}"
    )

]).partial(
    persona_instructions=persona_instructions
)


# =====================================================
# DOCUMENT CHAIN
# =====================================================

qa_chain = create_stuff_documents_chain(

    llm,

    qa_prompt

)


# =====================================================
# RAG CHAIN
# =====================================================

rag_chain = create_retrieval_chain(

    history_retriever,

    qa_chain

)

# =====================================================
# CHAT MEMORY
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# SHOW PREVIOUS MESSAGES
# =====================================================

for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.markdown(message.content)

    else:

        with st.chat_message("assistant"):

            st.markdown(message.content)


# =====================================================
# CHAT INPUT
# =====================================================

user_query = st.chat_input("Ask a question from your knowledge base...")

if user_query:

    # ---------- Show User Message ----------

    with st.chat_message("user"):

        st.markdown(user_query)

    st.session_state.chat_history.append(
        HumanMessage(content=user_query)
    )

    # ---------- AI Response ----------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = rag_chain.invoke({

                    "input": user_query,

                    "chat_history": st.session_state.chat_history

                })

                answer = response["answer"]

                st.markdown(answer)

            except Exception as e:

                answer = f"❌ Error: {str(e)}"

                st.error(answer)

    # ---------- Save AI Message ----------

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )


# =====================================================
# SIDEBAR TOOLS
# =====================================================

with st.sidebar:

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    if st.button("♻️ Rebuild Knowledge Base"):

        if os.path.exists("./chroma_db"):

            shutil.rmtree("./chroma_db")

        st.cache_resource.clear()

        st.success("Knowledge Base Rebuilt Successfully ✅")

        st.rerun()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "⚡ Titan RAG Chatbot • Powered by Groq + LangChain + ChromaDB"
)

