# fullstack gpt code challenge 08
import json
import streamlit as st
from duckduckgo_search import DDGS
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.schema import SystemMessage
from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from typing import Type


st.set_page_config(
    page_title="::: Research Agent :::",
    page_icon="📜",
)
st.title("Research Agent")

st.markdown(
"""
    Welcome to Research Agent!\n
    Use this chatbot to research somthing you're curious about.\n
    ex) Research about the XZ backdoor
"""
)
st.divider()


if "messages" not in st.session_state:
    st.session_state["messages"] = []

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


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


class SearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for information. ex) XZ backdoor"
    )

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
        Use this tool to search information on wikipedia site.
        It takes a query as an argument.
    """
    args_schema: Type[
        SearchToolArgsSchema
    ] = SearchToolArgsSchema

    def _run(self, query):
        wrapper = WikipediaAPIWrapper()
        search = WikipediaQueryRun(api_wrapper=wrapper)
        return search.run(query)

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
        Use this tool to search information on duckduckgo site.
        It takes a query as an argument.
    """
    args_schema: Type[
        SearchToolArgsSchema
    ] = SearchToolArgsSchema

    def _run(self, query):
        # fix : DuckDuckGoSearchAPIWrapper (HTTP Error) -> duckduckgo_search.DDGS
        # ddgs text -k "Research about the XZ backdoor"
        search = DDGS().text(query)
        return json.dumps(list(search))


class SearchResultParseToolArgsSchema(BaseModel):
    link: str = Field(
        description="The site link retrieved from web search"
    )

class SearchResultParseTool(BaseTool):
    name = "SearchResultParseTool"
    description = """
        Use this tool to load link to return detail content.
        It takes a link as an argument.
    """
    args_schema: Type[
        SearchResultParseToolArgsSchema
    ] = SearchResultParseToolArgsSchema

    def _run(self, link):
        loader = WebBaseLoader(link, verify_ssl=True)
        data = loader.load()
        return data


class SearchResultSaveToolArgsSchema(BaseModel):
    content: str = Field(
        description="The search result on wikipedia or duckduckgo site"
    )

class SearchResultSaveTool(BaseTool):
    name = "SearchResultSaveTool"
    description = """
        Use this tool to save web search result.
        It takes a result as an argument.
    """
    args_schema: Type[
        SearchResultSaveToolArgsSchema
    ] = SearchResultSaveToolArgsSchema

    def _run(self, content):
        file_path = "./challenge-08.result"
        with open(file_path, "w+", encoding="utf-8") as f:
            f.write(content)


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
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        )
    )

    # Github Repo Link
    st.markdown("---")
    github_link="https://github.com/toweringcloud/fullstack-gpt/blob/main/challenge-09.py"
    badge_link="https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


def main():
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar.")
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

    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        tools=[
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
            SearchResultParseTool(),
            SearchResultSaveTool(),
        ],
        agent_kwargs={
            "system_message": SystemMessage(
                content="""
                    You are a web research expert.

                    You search information by query and save the result contents into file.
                    Be sure to use two sites and summarize the results less than 1000 words.
                    If communication error occurs, skip the task and go to next step, please.
                """
            )
        },
    )

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    question = st.chat_input("Ask anything you're curious about.")
    if question:
        send_message(question, "human")

        with st.chat_message("ai"):
            st.markdown("Researching about your question...")
            agent.invoke(question)

    else:
        st.session_state["messages"] = []
        return

try:
    main()

except Exception as e:
    st.write(e)
