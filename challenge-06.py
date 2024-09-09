# fullstack gpt code challenge 06
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path


st.set_page_config(
    page_title="::: Quiz GPT :::",
    page_icon="🧐",
)

st.title("Quiz GPT")


# extension feature
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = PromptTemplate.from_template(
"""
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 5 questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    The difficulty level of the problem is '{level}'.

    Context: {context}
"""
)


@st.cache_data(show_spinner="Loading your file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/gpt_quiz/{file.name}"
    Path("./.cache/gpt_quiz").mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb+") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic, level):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "level": level})


@st.cache_data(show_spinner="Making Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None

    # API Key 입력
    openai_api_key = st.text_input("Input your OpenAI API Key")

    # AI Model 선택
    st.markdown("---")
    selected_model = st.selectbox(
        "Choose your AI Model",
        (
            "gpt-3.5-turbo",
            "gpt-4o-mini"
        )
    )

    # Quiz Target 선택 (Wiki or Custom File)
    st.markdown("---")
    quiz_target = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if quiz_target == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file.", type=["pdf", "txt", "docx"]
        )
        if file:
            docs = split_file(file)

    # Wiki를 선택했을 경우, Quiz Keyword 검색
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

    # Quiz Level 선택
    st.markdown("---")
    level = st.selectbox("Quiz Level", ("Easy", "Medium", "Hard"))

    # Github Repo Link
    st.markdown("---")
    st.write("[Github] https://github.com/toweringcloud/fullstack-gpt/blob/main/challenge-06.py")


if not docs:
    st.markdown(
    """
        Welcome to QuizGPT.
        I will make a quiz from Wikipedia or your own file to test your knowledge.               
        Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=selected_model,
            temperature=0.1,
            streaming=True,
            callbacks=[
                StreamingStdOutCallbackHandler(),
            ],
        ).bind(
            function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
        )

        response = run_quiz_chain(docs, topic if topic else file.name, level)
        response = response.additional_kwargs["function_call"]["arguments"]

        with st.form("questions_form"):
            questions = json.loads(response)["questions"]
            question_count = len(questions)
            success_count = 0

            for idx, question in enumerate(questions):
                st.markdown(f'#### {idx+1}. {question["question"]}')
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )

                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    success_count += 1
                elif value is not None:
                    st.error("Wrong!")
            if question_count == success_count:
                st.balloons()

            button = st.form_submit_button()
