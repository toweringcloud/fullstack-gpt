# fullstack gpt code challenge 09
import streamlit as st
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override


st.set_page_config(
    page_title="::: Research Assistant :::",
    page_icon="📜",
)
st.title("Research Assistant")

st.markdown(
"""
    Welcome to Research Assistant!\n
    Use this chatbot to research somthing you're curious about.\n
    ex) Research about the global warming
"""
)
st.divider()


# https://platform.openai.com/docs/assistants/tools/function-calling?context=streaming
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
            if tool.function.name == "WikipediaSearchTool":
                tool_outputs.append({"tool_call_id": tool.id, "output": "wiki"})
            elif tool.function.name == "DuckDuckGoSearchTool":
                tool_outputs.append({"tool_call_id": tool.id, "output": "ddg"})

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
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
        self.save_final_result(self.message)

    def save_research_result(self, content):
        file_path = "./challenge-09.result"
        with open(file_path, "w+", encoding="utf-8") as f:
            f.write(content)
        st.markdown(f"research result saved at {file_path}")


function_wiki = {
    "type": "function",
    "function": {
        "name": "WikipediaSearchTool",
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
}

function_ddgs = {
    "type": "function",
    "function": {
        "name": "DuckDuckGoSearchTool",
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
}


with st.sidebar:
    # API Key 입력
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
        type="password"
    )

    # AI Model 선택
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
        assistant = client.beta.assistants.create(
            name="Research Expert",
            instructions="You are a web research bot. Search information by query and save summarized result into file.",
            temperature=0.1,
            model=selected_model,
            tools=[
                function_wiki,
                function_ddgs,
            ],
        )
        thread = client.beta.threads.create()

        st.session_state["client"] = client
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread

        with st.chat_message("ai"):
            st.markdown("I'm ready! Ask away!")

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if messages:
        messages = list(messages)
        messages.reverse()
        for message in messages:
            st.chat_message(message.role).write(message.content[0].text.value)

    question = st.chat_input("Ask anything you're curious about.")
    if question:
        with st.chat_message("human"):
            st.markdown(question)

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="human",
            contents=question
        )

        with st.chat_message("ai"):
            st.markdown("Researching about your question...")

            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler()
            ) as stream:
                stream.until_done()

try:
    main()

except Exception as e:
    st.write(e)
