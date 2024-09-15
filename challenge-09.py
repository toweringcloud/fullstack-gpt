# fullstack gpt code challenge 09
import json
import streamlit as st
from duckduckgo_search import DDGS
from langchain.document_loaders import WebBaseLoader
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override


# show main ui
st.set_page_config(
    page_title="::: Research Assistant :::",
    page_icon="📜",
)
st.title("Research Assistant")

st.markdown(
"""
    Welcome to Research Assistant!\n
    Use this chatbot to research somthing you're curious about.\n
"""
)
st.divider()


# show sidebar ui
with st.sidebar:
    # API Key
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
        type="password"
    )

    # AI Model
    selected_model = st.selectbox(
        "Choose your AI Model",
        (
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        )
    )

    # Github Repo
    st.markdown("---")
    github_link="https://github.com/toweringcloud/fullstack-gpt/blob/main/challenge-09.py"
    badge_link="https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


# define function logic
def WikipediaSearchTool(params):
    query = params["query"]
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return search.invoke(query)

def DuckDuckGoSearchTool(params):
    query = params["query"]
    search = DDGS().text(query)
    return json.dumps(list(search))

def SearchResultParseTool(params):
    link = params["link"]
    loader = WebBaseLoader(link, verify_ssl=True)
    return loader.load()

# define function mapper
functions_map = {
    "wiki_search": WikipediaSearchTool,
    "ddg_search": DuckDuckGoSearchTool,
    "link_parse": SearchResultParseTool,
}

# define function schema
functions = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search information on wikipedia site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user's input",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ddg_search",
            "description": "Search information on duckduckgo site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user's input",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_parse",
            "description": "Load link to parse into detail content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "link": {
                        "type": "string",
                        "description": "search result's output",
                    },
                },
                "required": ["link"],
            },
        },
    }
]


# https://platform.openai.com/docs/assistants/tools/function-calling?context=streaming
# assistant event handler with streaming
class EventHandler(AssistantEventHandler):
    message = ""

    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == 'thread.run.requires_action':
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.message_box = st.empty()
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            print(f"# tool: {tool.id} | {tool.function.name} | {tool.function.arguments}")
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": functions_map[tool.function.name](json.loads(tool.function.arguments))
            })

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        # NameError: name 'client' is not defined <- stream.until_done()
        client = st.session_state["client"]

        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                self.message += text
                self.message_box.markdown(self.message)
                print(text, end="", flush=True)
            print()
            self.save_research_result()

    def save_research_result(self):
        st.session_state["result"] = self.message
        file_path = "./challenge-09.result"
        with open(file_path, "w+", encoding="utf-8") as f:
            f.write(self.message)
        st.markdown(f"research result saved at {file_path}")


def main():
    client = None

    if "client" in st.session_state:
        client = st.session_state["client"]
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    else:
        if not openai_api_key:
            st.error("Please input your OpenAI API Key on the sidebar.")
            return

        # https://pypi.org/project/openai
        client = OpenAI(api_key=openai_api_key)

        # BadRequestError: Error code: 400 - {'error': {'message': "Unknown parameter: 'tools[1].function. '.", 'type': 'invalid_request_error', 'param': 'tools[1].function. ', 'code': 'unknown_parameter'}}
        assistant = client.beta.assistants.create(
            name="Research Expert",
            instructions="You are a web research bot. Search information by query, parse web links and summarize the result.",
            temperature=0.1,
            model=selected_model,
            tools=functions,
        )
        thread = client.beta.threads.create()

        st.session_state["client"] = client
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread

        with st.chat_message("ai"):
            st.markdown("I'm ready! Ask away!")

    # show messages in the thread of your assistant
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if messages:
        messages = list(messages)
        messages.reverse()
        for message in messages:
            st.chat_message(message.role).write(message.content[0].text.value)

    # ready to research your question
    question = st.chat_input("Ask anything you're curious about.")
    if question:
        with st.chat_message("human"):
            st.markdown(question)

        # BadRequestError: Error code: 400 - {'error': {'message': "Invalid value: 'human'. Supported values are: 'user' and 'assistant'.", 'type': 'invalid_request_error', 'param': 'role', 'code': 'invalid_value'}}
        # BadRequestError: Error code: 400 - {'error': {'message': "Can't add messages to thread_i2LyHXXF8dZGKOy0zTIe2Riq while a run run_cVzucEYxgiNsR9rSPmFIHBJG is active.", 'type': 'invalid_request_error', 'param': None, 'code': None}
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )

        with st.chat_message("ai"):
            st.markdown("Researching about your question...")

            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler()
            ) as stream:
                stream.until_done()
    else:
        st.empty()

try:
    main()

except Exception as e:
    st.write(e)
