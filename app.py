import os
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain.chains.combine_documents import create_stuff_documents_chain


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Titan Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM UI ----------------

st.markdown("""
<style>

/* Background */
.stApp {
    background-color:#0e1117;
}


/* Heading */
h1 {
    color:white !important;
}


.stCaption {
    color:#cbd5e1 !important;
}


/* Chat bubbles */
[data-testid="stChatMessage"] {

    background-color:#1e293b !important;
    border-radius:14px;
    padding:12px;

}


/* Chat text */
[data-testid="stChatMessage"] p {

    color:white !important;
    font-size:16px;

}


/* User input box */
[data-testid="stChatInput"] {

    background-color:#1e293b !important;
    border:1px solid #475569 !important;
    border-radius:14px;

}


/* Input typed text */
[data-testid="stChatInput"] textarea {

    color:white !important;
    caret-color:white !important;
    font-size:16px !important;

}


/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder {

    color:#94a3b8 !important;

}


/* Send button */
[data-testid="stChatInputSubmitButton"] button {

    background:#ff5b5b !important;
    color:white !important;
    border-radius:10px;

}


/* Sidebar */
section[data-testid="stSidebar"] {

    background:#111827;

}


</style>
""", unsafe_allow_html=True)



st.title("⚡ My Digital Twin Chatbot")

st.caption(
    "A RAG-powered assistant trained to talk, think, and respond just like me."
)



# ---------------- API KEY ----------------


try:

    secret_groq_key = st.secrets["GROQ_API_KEY"]

except:

    secret_groq_key = None



with st.sidebar:

    st.header("⚙️ Configuration")


    if secret_groq_key:

        st.success("Groq API loaded 🔒")

        api_key = secret_groq_key


    else:

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_..."
        )


    st.divider()


    persona_instructions = st.text_area(

        "Persona Rules",

        value="""
1. Speak casually with confidence.
2. Keep answers concise.
3. Use natural language.
4. Use retrieved context when answering.
""",

        height=150

    )



if not api_key:

    st.info(
        "Add your Groq API key in sidebar or Streamlit secrets.",
        icon="🔑"
    )

    st.stop()



# ---------------- VECTOR DATABASE ----------------


@st.cache_resource(show_spinner="Building knowledge base...")


def initialize_vector_store():


    data_path="./data"
    db_path="./chroma_db"



    if not os.path.exists(data_path):

        os.makedirs(data_path)



    files=os.listdir(data_path)



    if not files:

        with open(
            f"{data_path}/about_me.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "I am an innovative developer passionate about AI, automation and modern web applications."
            )



    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )



    if os.path.exists(db_path):


        vectorstore = Chroma(

            persist_directory=db_path,

            embedding_function=embeddings

        )


    else:


        loader = DirectoryLoader(

            data_path,

            glob="*.txt",

            loader_cls=TextLoader,

            loader_kwargs={
                "encoding":"utf-8"
            }

        )


        documents = loader.load()



        splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=50

        )


        splits = splitter.split_documents(documents)



        vectorstore = Chroma.from_documents(

            documents=splits,

            embedding=embeddings,

            persist_directory=db_path

        )



    return vectorstore.as_retriever(

        search_kwargs={"k":3}

    )



retriever = initialize_vector_store()



# ---------------- GROQ MODEL ----------------


llm = ChatGroq(

    groq_api_key=api_key,

    model_name="llama-3.3-70b-versatile",

    temperature=0.7

)



# ---------------- RAG ----------------


contextualize_prompt = ChatPromptTemplate.from_messages([

    (
        "system",

        """
Given chat history and latest question,
rewrite the question clearly.
"""
    ),

    MessagesPlaceholder(
        "chat_history"
    ),

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



system_prompt = """

You are a digital twin.

Follow these rules:

{persona_instructions}


Use this knowledge:

{context}

"""



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



qa_chain=create_stuff_documents_chain(

    llm,

    qa_prompt

)



rag_chain=create_retrieval_chain(

    history_retriever,

    qa_chain

)



# ---------------- MEMORY ----------------


if "chat_history" not in st.session_state:

    st.session_state.chat_history=[]



for msg in st.session_state.chat_history:


    role="user" if isinstance(msg,HumanMessage) else "assistant"


    with st.chat_message(role):

        st.write(msg.content)



# ---------------- CHAT ----------------


if user_query := st.chat_input("Ask me anything..."):


    with st.chat_message("user"):

        st.write(user_query)



    with st.chat_message("assistant"):


        with st.spinner("Thinking..."):


            response = rag_chain.invoke({

                "input":user_query,

                "chat_history":st.session_state.chat_history

            })


            answer=response["answer"]


            st.write(answer)



    st.session_state.chat_history.append(

        HumanMessage(content=user_query)

    )


    st.session_state.chat_history.append(

        AIMessage(content=answer)

    )
