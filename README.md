# fullstack-gpt

langchain based gpt & agent service

## code challenge list

### challenge-01 : ChatPromptTemplate + LCEL

### challenge-02 : FewShotChatMessagePromptTemplate

### challenge-03 : ConversationSummaryBufferMemory + RunnablePassthrough

### challenge-04 : RAG Pipeline with a stuff document chain

### challenge-05 : Document GPT with langchain & streamlit

## how to run

### setup

-   install python 3.11.6 and add system path on python & pip

```
$ python --version
Python 3.11.6

$ pip --version
pip 23.2.1 from D:\setup\Python311\Lib\site-packages\pip (python 3.11)

```

-   install required packages

```
$ pip install -r requirements.txt
```

### config

-   set runtime environment

```
$ cat ./env
OPENAI_API_KEY="..."
```

-   load runtime environment

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

-   run normal app in virtual environment

```
$ python -m venv ./env
$ source ./env/Scripts/activate or source env/bin/activate
$ vi main.py
$ python main.py
$ deactivate
```

-   run jupyter app in virtual environment

```
$ python -m venv ./env
$ source ./env/Scripts/activate or source env/bin/activate
$ touch main.ipynb
! select runtime kernel as venv - python 3.11.6
! run code & debug for testing
$ deactivate
```

-   run streamlit app in root environment

```
$ streamlit run challenge-06.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```
