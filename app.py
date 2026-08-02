import os
import streamlit as st

from langchain_community.document_loaders import (
    DirectoryLoader,
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



# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Titan Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)



# ---------------- UI CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color:#0e1117;
}


.stChatMessage {

    background-color:#1e293b;
    color:white;
    border-radius:12px;
    padding:10px;

}


.stChatMessage p {

    color:white !important;

}


[data-testid="stChatInput"] textarea {

    color:white !important;

}


[data-testid="stChatInput"] {

    background-color:#1e293b;

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



    # -------- PDF UPLOAD --------


    st.divider()


    st.subheader("📄 Upload Knowledge PDF")


    uploaded_pdf = st.file_uploader(

        "Upload PDF",

        type=["pdf"]

    )


    if uploaded_pdf:


        upload_folder = "data/uploads"


        os.makedirs(

            upload_folder,

            exist_ok=True

        )


        pdf_location = f"{upload_folder}/{uploaded_pdf.name}"



        with open(pdf_location,"wb") as f:

            f.write(
                uploaded_pdf.getbuffer()
            )


        st.success(
            "PDF uploaded successfully!"
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



@st.cache_resource(
    show_spinner="Building knowledge base..."
)


def initialize_vector_store():


    data_path = "./data"

    pdf_path = "./data/uploads"

    db_path = "./chroma_db"



    os.makedirs(
        data_path,
        exist_ok=True
    )


    os.makedirs(
        pdf_path,
        exist_ok=True
    )



    if not os.listdir(data_path):


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




    # If database already exists

    if os.path.exists(db_path):


        vectorstore = Chroma(

            persist_directory=db_path,

            embedding_function=embeddings

        )



    else:



        documents = []



        # TXT files


        loader = DirectoryLoader(

            data_path,

            glob="*.txt",

            loader_cls=TextLoader,

            loader_kwargs={

                "encoding":"utf-8"

            }

        )



        documents.extend(

            loader.load()

        )




        # PDF files


        if os.path.exists(pdf_path):


            for file in os.listdir(pdf_path):


                if file.endswith(".pdf"):


                    pdf_loader = PyPDFLoader(

                        f"{pdf_path}/{file}"

                    )


                    documents.extend(

                        pdf_loader.load()

                    )




        splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=50

        )



        splits = splitter.split_documents(

            documents

        )



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





# ---------------- GROQ MODEL ----------------



llm = ChatGroq(

    groq_api_key=api_key,

    model_name="llama-3.3-70b-versatile",

    temperature=0.7

)






# ---------------- RAG CHAIN ----------------



contextualize_prompt = ChatPromptTemplate.from_messages([


    (

        "system",

        """

Given the chat history and latest user question,
rewrite the question so it can be understood independently.

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

Follow these personality rules:

{persona_instructions}


Use this retrieved knowledge:

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






# ---------------- CHAT MEMORY ----------------



if "chat_history" not in st.session_state:


    st.session_state.chat_history=[]





for msg in st.session_state.chat_history:


    role = (

        "user"

        if isinstance(msg,HumanMessage)

        else "assistant"

    )


    with st.chat_message(role):

        st.write(
            msg.content
        )






# ---------------- CHAT ----------------



if user_query := st.chat_input(

    "Ask me anything..."

):


    with st.chat_message("user"):


        st.write(
            user_query
        )




    with st.chat_message("assistant"):


        with st.spinner(

            "Thinking..."

        ):



            response = rag_chain.invoke({


                "input":user_query,


                "chat_history":st.session_state.chat_history


            })



            answer=response["answer"]



            st.write(

                answer

            )





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
