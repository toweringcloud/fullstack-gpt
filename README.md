# fullstack-gpt

langchain based gpt & agent service

## code challenge list

### challenge-01 : ChatPromptTemplate + LCEL

### challenge-02 : FewShotChatMessagePromptTemplate

### challenge-03 : ConversationSummaryBufferMemory + RunnablePassthrough

### challenge-04 : RAG pipeline with a stuff document chain

### challenge-05 : Document GPT with streamlit app + RunnableLambda

- [demo] https://fullstack-gpt-document.streamlit.app

### challenge-06 : Quiz GPT with custom function calling

- [demo] https://fullstack-gpt-quiz.streamlit.app

### challenge-07 : Site GPT with map re-rank chain

- [demo] https://fullstack-gpt-site.streamlit.app

### challenge-08 : Research Agent with wikipedia or duckduckgo

### challenge-09 : Research Assistant with agent + thread + run

### challenge-10 : Multiple Research Staffs with Crew AI (TBD)

### challenge-11 : Meeting GPT with refined chain

- [demo] https://fullstack-gpt-meeting.streamlit.app

### challenge-12 : Private GPT with Ollama LLM

### challenge-13 : Chef GPT for custom ChatGTPs action (TBD)

## how to run

### setup

- install python 3.11.6 and add system path on python & pip

```
$ python --version
Python 3.11.6

$ pip --version
pip 23.2.1 from D:\setup\Python311\Lib\site-packages\pip (python 3.11)

```

- install required packages

```
$ pip install -r requirements.txt
```

### config

- set runtime environment

```
$ cat .env
OPENAI_API_KEY="..."
```

- load runtime environment

```
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI

config = dotenv_values(".env")

llm = ChatOpenAI(
    openai_api_key=config['OPENAI_API_KEY'],
    ...
)
```

### launch

- run normal app in virtual environment

```
$ python -m venv ./env
$ source env/Scripts/activate or source env/bin/activate
$ vi main.py
$ python main.py
$ deactivate
```

- run jupyter app in virtual environment

```
$ source env/Scripts/activate or source env/bin/activate
$ touch main.ipynb
! select runtime kernel as venv - python 3.11.6
! run code & debug for testing
$ deactivate
```

- run streamlit app in root environment

```
$ streamlit run challenge-06.py
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### test with headless browser

```
$ source env/Scripts/activate or source env/bin/activate
$ playwright install
Downloading Chromium 119.0.6045.9 (playwright build v1084) from https://playwright.azureedge.net/builds/chromium/1084/chromium-win64.zip
...
|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■| 100% of 120.8 Mb
Chromium 119.0.6045.9 (playwright build v1084) downloaded to ...

Downloading FFMPEG playwright build v1009 from https://playwright.azureedge.net/builds/ffmpeg/1009/ffmpeg-win64.zip
...
|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■| 100% of 1.4 Mb
FFMPEG playwright build v1009 downloaded to ...

Downloading Firefox 118.0.1 (playwright build v1425) from https://playwright.azureedge.net/builds/firefox/1425/firefox-win64.zip
...
|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■| 100% of 80 Mb
Firefox 118.0.1 (playwright build v1425) downloaded to ...

Downloading Webkit 17.4 (playwright build v1921) from https://playwright.azureedge.net/builds/webkit/1921/webkit-win64.zip
...
|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■| 100% of 44.5 Mb
Webkit 17.4 (playwright build v1921) downloaded to ...
```
