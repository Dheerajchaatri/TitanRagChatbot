# PART 1

import os
import shutil
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

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



# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Titan Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)



# ================= UI CSS =================

st.markdown("""
<style>

.stApp {
    background-color:#0e1117 !important;
}


/* Title */

h1 {
    color:white !important;
}


/* Caption */

.stCaption {
    color:#cbd5e1 !important;
}


/* Chat messages */

[data-testid="stChatMessage"] {

    background-color:#1e293b !important;

    border-radius:14px !important;

    padding:12px !important;

}


[data-testid="stChatMessage"] p {

    color:white !important;

}


/* Chat input */

[data-testid="stChatInput"] {

    background-color:#111827 !important;

    border:1px solid #475569 !important;

    border-radius:14px !important;

}


[data-testid="stChatInput"] textarea {

    background-color:#111827 !important;

    color:white !important;

    -webkit-text-fill-color:white !important;

    caret-color:white !important;

}


[data-testid="stChatInput"] textarea::placeholder {

    color:#94a3b8 !important;

}


/* Sidebar */

section[data-testid="stSidebar"] {

    background:#111827 !important;

}


</style>

""", unsafe_allow_html=True)



st.title("⚡ My Digital Twin Chatbot")

st.caption(
    "A RAG-powered assistant trained to talk, think, and respond just like me."
)



# ================= API KEY =================


try:

    secret_groq_key = st.secrets["GROQ_API_KEY"]

except:

    secret_groq_key = None



with st.sidebar:


    st.header("⚙️ Configuration")



    if secret_groq_key:


        st.success(
            "Groq API loaded 🔒"
        )

        api_key = secret_groq_key



    else:


        api_key = st.text_input(

            "Groq API Key",

            type="password",

            placeholder="gsk_..."

        )



    st.divider()



    # ================= PDF UPLOAD =================


    st.subheader(
        "📄 Upload Knowledge PDF"
    )


    uploaded_file = st.file_uploader(

        "Upload PDF file",

        type=["pdf"]

    )



    if uploaded_file:


        upload_folder = "./data/uploads"


        os.makedirs(

            upload_folder,

            exist_ok=True

        )



        pdf_path = os.path.join(

            upload_folder,

            uploaded_file.name

        )



        with open(

            pdf_path,

            "wb"

        ) as f:


            f.write(

                uploaded_file.getbuffer()

            )



        # Remove old database so new PDF gets indexed

        if os.path.exists("./chroma_db"):

            shutil.rmtree("./chroma_db")



        st.cache_resource.clear()



        st.success(

            "PDF uploaded successfully ✅"

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




# Stop if API missing

if not api_key:


    st.info(

        "Add your Groq API key in sidebar or Streamlit secrets.",

        icon="🔑"

    )


    st.stop()

# PART 2


# ================= VECTOR DATABASE =================


@st.cache_resource(show_spinner="Building knowledge base...")


def initialize_vector_store():


    data_path = "./data"

    upload_path = "./data/uploads"

    db_path = "./chroma_db"



    # Create folders

    os.makedirs(

        data_path,

        exist_ok=True

    )


    os.makedirs(

        upload_path,

        exist_ok=True

    )



    # Find existing files

    documents_files = []



    for root, dirs, files in os.walk(data_path):


        for file in files:


            if file.endswith(".txt") or file.endswith(".pdf"):


                documents_files.append(

                    os.path.join(root, file)

                )



    # If no file exists create default knowledge

    if not documents_files:


        default_file = os.path.join(

            data_path,

            "about_me.txt"

        )


        with open(

            default_file,

            "w",

            encoding="utf-8"

        ) as f:


            f.write(

                "I am an innovative developer passionate about AI, automation and modern web applications."

            )



        documents_files.append(default_file)





    # Embeddings


    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )




    # Load existing Chroma DB

    if os.path.exists(db_path):


        vectorstore = Chroma(

            persist_directory=db_path,

            embedding_function=embeddings

        )



    else:



        documents = []



        # Read files


        for file_path in documents_files:



            # TXT files

            if file_path.endswith(".txt"):



                loader = TextLoader(

                    file_path,

                    encoding="utf-8"

                )


                documents.extend(

                    loader.load()

                )




            # PDF files


            elif file_path.endswith(".pdf"):



                loader = PyPDFLoader(

                    file_path

                )


                documents.extend(

                    loader.load()

                )





        # Split text into chunks


        splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=50

        )



        splits = splitter.split_documents(

            documents

        )





        # Create vector database


        vectorstore = Chroma.from_documents(

            documents=splits,

            embedding=embeddings,

            persist_directory=db_path

        )





    return vectorstore.as_retriever(

        search_kwargs={

            "k":3

        }

    )





retriever = initialize_vector_store()

# PART 3


# ================= GROQ MODEL =================


llm = ChatGroq(

    groq_api_key=api_key,

    model_name="llama-3.3-70b-versatile",

    temperature=0.7

)



# ================= RAG PIPELINE =================


contextualize_prompt = ChatPromptTemplate.from_messages([


    (

        "system",

        """
Given the chat history and latest user question,
rewrite the question so it can be understood independently.
Only use conversation context.
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

You are a helpful AI assistant.

Answer the user using only the provided knowledge context.

If information is not available in the uploaded documents,
clearly say that you don't have that information.


Personality rules:

{persona_instructions}


Knowledge context:

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





qa_chain = create_stuff_documents_chain(

    llm,

    qa_prompt

)





rag_chain = create_retrieval_chain(

    history_retriever,

    qa_chain

)





# ================= CHAT MEMORY =================


if "chat_history" not in st.session_state:


    st.session_state.chat_history = []





# Show previous messages


for msg in st.session_state.chat_history:


    role = (

        "user"

        if isinstance(msg, HumanMessage)

        else "assistant"

    )


    with st.chat_message(role):


        st.write(msg.content)





# ================= CHAT INPUT =================


if user_query := st.chat_input(

    "Ask me anything..."

):



    # User message


    with st.chat_message("user"):


        st.write(user_query)





    # AI response


    with st.chat_message("assistant"):


        with st.spinner("Thinking..."):



            response = rag_chain.invoke({


                "input": user_query,


                "chat_history": st.session_state.chat_history


            })



            answer = response["answer"]



            st.write(answer)





    # Save chat history


    st.session_state.chat_history.append(

        HumanMessage(

            content=user_query

        )

    )



    st.session_state.chat_history.append(

        AIMessage(

            content=answer

        )

    )
