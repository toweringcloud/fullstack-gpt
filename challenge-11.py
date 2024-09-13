# fullstack gpt code challenge 11
import glob
import math
import os
import streamlit as st
import subprocess
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from openai import OpenAI
from pathlib import Path
from pydub import AudioSegment
from pytubefix import YouTube


st.set_page_config(
    page_title="::: Meeting GPT :::",
    page_icon="💼",
)
st.title("Meeting GPT")

st.markdown(
"""
    Welcome to Meeting GPT!

    Upload a video and I will give you a transcript, a summary and a chatbot to ask any questions about it.

    Get started by uploading a video file in the sidebar.
"""
)


def has_transcript():
    fileExist = os.path.exists(transcript_path)
    fileEmpty = Path(transcript_path).stat().st_size == 0
    return fileExist and not fileEmpty


@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcript():
        return
    if not os.path.exists(video_path):
        print(f"video({video_path}) not available!")

    # check ffmpeg utility installed
    """
        $ choco install ffmpeg -y
        $ ffmpeg -version
        ffmpeg version 7.0.2-essentials_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
        built with gcc 13.2.0 (Rev5, Built by MSYS2 project)
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript():
        return
    if not os.path.exists(audio_path):
        print(f"audio({video_path}) not available!")

    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


@st.cache_data()
def transcribe_chunks(chunks_folder, destination):
    if has_transcript():
        return

    print(f"transcribe_chunks.i: {chunks_folder} | {destination}")
    files = glob.glob(f"{chunks_folder}/*.mp3")
    files.sort()
    print(f"transcribe_chunks.c: {len(files)} files\n")

    # https://pypi.org/project/openai
    client = OpenAI(api_key=openai_api_key)

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # https://platform.openai.com/docs/guides/speech-to-text
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
            print(transcription.text)
            text_file.write(transcription["text"])
    print(f"transcribe_chunks.o: {Path(destination).stat().st_size} bytes")


# @st.cache_data()
@st.cache_resource()
def embed_file(file_path):
    dir_path = f"./.cache/embeddings/{video_name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    cache_dir = LocalFileStore(dir_path)
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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
            "gpt-3.5-turbo",
            "gpt-4o-mini"
        )
    )

    # Video 파일 업로드
    video_source = st.file_uploader(
        "Upload your Video file",
        type=["mp4", "avi", "mkv", "mov"],
    )

    # Github Repo Link
    st.markdown("---")
    github_link="https://github.com/toweringcloud/fullstack-gpt/blob/main/challenge-11.py"
    badge_link="https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")

if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")

else:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=selected_model,
        temperature=0.1,
    )

if video_source:
    video_name = video_source.name
    video_extension = Path(video_name).suffix
    video_path = f"./files/{video_name}"
    audio_path = video_path.replace(video_extension, ".mp3")
    transcript_path = video_path.replace(video_extension, ".txt")
    chunks_path = f"./.cache/chunks/{video_name}"
    Path(chunks_path).mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )

    with st.status("Loading video...") as status:
        video_content = video_source.read()
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path, audio_path)

        status.update(label="Cutting audio segments...")
        chunk_minutes = 3
        cut_audio_in_chunks(audio_path, chunk_minutes, chunks_path)

        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_path, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")
        if start:
            if has_transcript():
                loader = TextLoader(transcript_path)
                docs = loader.load_and_split(text_splitter=splitter)

                first_summary_prompt = ChatPromptTemplate.from_template(
                """
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:
                """
                )

                first_summary_chain = first_summary_prompt | llm | StrOutputParser()
                summary = first_summary_chain.invoke(
                    {"text": docs[0].page_content},
                )

                refine_prompt = ChatPromptTemplate.from_template(
                """
                    Your job is to produce a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ------------
                    {context}
                    ------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                """
                )

                refine_chain = refine_prompt | llm | StrOutputParser()

                with st.status("Summarizing...") as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                        summary = refine_chain.invoke(
                            {
                                "existing_summary": summary,
                                "context": doc.page_content,
                            }
                        )
                        st.write(summary)
                st.write(summary)

            else:
                print(f"{transcript_path} not available!")

    with qa_tab:
        if has_transcript():
            retriever = embed_file(transcript_path)
            question = st.text_input("Input question about your audio script.")

            if question:
                docs = retriever.invoke(question)
                for doc in docs:
                    st.write(f"- {doc.page_content}")
        else:
            print(f"{transcript_path} not available!")

else:
    # [Nomadcoders] What is the Difference Between SQLite, MySQL and PostgreSQL?
    # https://github.com/conf42/src/commit/7326e47af48bf61e7e279136006589cc75b940f0
    # pip install pytubefix
    sample_video = "http://youtube.com/watch?v=ocZid4g4UpY"
    download_ok = False
    if download_ok:
        yt = YouTube(sample_video)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path="./.cache", filename="podcast.mp4")
