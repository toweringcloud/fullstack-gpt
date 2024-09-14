# fullstack gpt code challenge 05
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pathlib import Path


st.set_page_config(
    page_title="::: Document GPT :::",
    page_icon="📜",
)
st.title("Document GPT")

st.markdown(
"""
Welcome to Document GPT!
            
Use this chatbot to ask questions to an AI about your files.

1. Input your OpenAI API Key on the sidebar
2. Upload your file on the sidebar.
3. Ask questions related to the document.
"""
)
st.divider()

with st.sidebar:
    # API Key 입력
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
        type="password"
    )

    # Model 선택
    selected_model = st.selectbox(
        "Choose your AI Model",
        (
            "gpt-3.5-turbo",
            "gpt-4o-mini",
        )
    )

    # File 선택
    file = st.file_uploader(
        "Upload a. txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    # Github Repo Link
    st.markdown("---")
    github_link="https://github.com/toweringcloud/fullstack-gpt/blob/main/challenge-05.py"
    badge_link="https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


# @st.cache_data(show_spinner="Embedding file...")
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/gpt_doc/{file.name}"
    Path("./.cache/gpt_doc").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)

    cache_path = "./.cache/gpt_doc/embeddings/"
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    cache_dir = LocalFileStore(cache_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. 
            If you don't know the answer just say you don't know. 
            DON't make anything up

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def main():
    if not openai_api_key:
        return

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=selected_model,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    if file:
        retriever = embed_file(file)

        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file.....")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)

    else:
        st.session_state["messages"] = []
        return

try:
    main()

except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)
