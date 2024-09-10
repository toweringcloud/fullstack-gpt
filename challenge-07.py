# fullstack gpt code challenge 07
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


st.set_page_config(
    page_title="::: Site GPT :::",
    page_icon="🖥️",
)
st.title("Site GPT")


answers_prompt = ChatPromptTemplate.from_template(
"""
    Using ONLY the following context, answer the user's question.
    If you can't just say you don't know, don't make anything up.
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                Use ONLY the following pre-existing answers to answer the user's question.
                Use the answers that have the highest score (more helpful) and favor the most recent ones.
                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"{answer['answer']}\n- Source:{answer['source']}\n- Date:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()

    footer = soup.find("footer")
    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        # .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            # r"^(.*\/products\/|.*\/case-studies\/).*",
            r"^(.*\/products\/|.*\/learning\/).*",
        ],
        parsing_function=parse_page,
        # blocksize=16380,
        continue_on_failure=True
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()


with st.sidebar:
    docs = None

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

    # SiteMap URL 입력
    st.markdown("---")
    sitemap_url = st.text_input(
        "Write down a SiteMap URL",
        placeholder="https://example.com",
    )


st.markdown(
"""
    Welcome to Site GPT.

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
"""
)

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
    )

if sitemap_url:
    if ".xml" not in sitemap_url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        # web.Application(handler_args={'max_field_size': 16380})
        # ClientResponseError: 400, message='Got more than 8190 bytes (12827) when reading Header value is too long.', url=URL('https://www.cloudflare.com/application-services/products/cloudflare-spectrum/')
        '''
            Fetching pages:   2%|1         | 39/1961 [00:03<02:34, 12.47it/s]Error fetching https://www.cloudflare.com/products/stream-delivery/, skipping due to continue_on_failure=True
            Fetching pages:   3%|2         | 53/1961 [00:04<02:27, 12.96it/s]Error fetching https://www.cloudflare.com/ecommerce/, skipping due to continue_on_failure=True
            Fetching pages:   3%|3         | 67/1961 [00:05<02:26, 12.96it/s]Error fetching https://www.cloudflare.com/performance/, skipping due to continue_on_failure=True
            Fetching pages:  20%|#9        | 383/1961 [00:58<04:19,  6.09it/s]Error fetching https://www.cloudflare.com/plans/pro/, skipping due to continue_on_failure=True
            Fetching pages:  22%|##1       | 424/1961 [01:08<06:43,  3.81it/s]Error fetching https://www.cloudflare.com/what-is-cloudflare/, skipping due to continue_on_failure=True
            Fetching pages:  40%|####      | 789/1961 [02:29<04:35,  4.26it/s]Error fetching https://www.cloudflare.com/partners/threat-intelligence/, skipping due to continue_on_failure=True
            Fetching pages:  70%|######9   | 1365/1961 [03:15<00:15, 37.74it/s]Error fetching https://www.cloudflare.com/case-studies/luana-savings-bank/, skipping due to continue_on_failure=True
            Fetching pages:  91%|#########1| 1794/1961 [06:02<01:30,  1.85it/s]Error fetching https://www.cloudflare.com/press-releases/2023/cloudflare-partners-with-databricks/, skipping due to continue_on_failure=True
            Fetching pages:  97%|#########6| 1894/1961 [06:57<00:34,  1.93it/s]Error fetching https://www.cloudflare.com/press-releases/2020/cloudflare-announces-date-of-third-quarter-2020-financial-results/, skipping due to continue_on_failure=True
            Fetching pages: 100%|##########| 1961/1961 [07:27<00:00,  4.39it/s]
            Retrying langchain.embeddings.openai.embed_with_retry.<locals>._embed_with_retry in 4.0 seconds as it raised RateLimitError: Rate limit reached for text-embedding-ada-002 in organization org-UvVnVP4ROYREanHJyUM7wz4O on tokens per min (TPM): Limit 1000000, Used 735993, Requested 747646. Please try again in 29.018s. Visit https://platform.openai.com/account/rate-limits to learn more..

            Fetching pages:  89%|########9 | 367/412 [02:00<00:14,  3.13it/s]Error fetching https://www.cloudflare.com/case-studies/okcupid/, skipping due to continue_on_failure=True
            Fetching pages: 100%|##########| 412/412 [02:17<00:00,  2.99it/s]
        '''
        retriever = load_website(sitemap_url)

        st.divider()
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
