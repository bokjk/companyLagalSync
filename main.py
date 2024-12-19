import streamlit as st

from langchain_core.messages.chat import ChatMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_teddynote.prompts import load_prompt
from langchain import hub

# 프롬프트 파일 목록 가져오기
import glob

import os

st.title("취업규칙 상담 시스템")


# 파일을 캐시에 저장 (시간이 오래 걸리는 작업을 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다... ")
def embed_file():

    # 1단계: 문서로드 ---------------------------------------------
    text_files = {
        "근로기준법": "./data/근로기준법(법률)(제20520호)(20241022).txt",
        "취업규칙": "./data/취업규칙_되고시스템 240610.txt",
    }
    documents = []
    for doc_type, file_path in text_files.items():
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 각 문서의 출처를 메타데이터로 추가
            documents.append({"content": content, "source": doc_type})

    # 2단계: 문서 분할 ---------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = []
    for doc in documents:
        splits = text_splitter.create_documents(
            texts=[doc["content"]], metadatas=[{"source": doc["source"]}]
        )
        split_documents.extend(splits)

    # 3단계: 임베딩 생성 ---------------------------------------------
    embeddings = OpenAIEmbeddings()

    # 4단계: DB생성 및 저장   ---------------------------------------------
    # 벡터 스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 5 단계 : 검색시(Retriever) 생성 ---------------------------------------------
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # 관련 문서 조각을 4개까지 검색
    )

    return retriever


# 체인 생성
def create_chain(retriever):
    # prompt | llm | output_parser
    # prompt = load_prompt("prompts/sns.yaml")
    # prompt = load_prompt(prompt_filepath)

    # 6단계 : 프롬프트 생성 ---------------------------------------------
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    # 7 단계 : 언어모델 생성 ---------------------------------------------
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 8 단계: 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 체인 생성
    return chain


# 맨처음 한번만 실행
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# chain을 저장
if "chain" not in st.session_state:
    # st.session_state["chain"] = None
    retriever = embed_file()
    chain = create_chain(retriever)
    st.session_state["chain"] = chain


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 대화를 저장하는 함수
def add_message(role: str, message: str):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:

    # chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        ## 스트림으로 답변
        response = chain.stream(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍출력한다.
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        # 대화를 저장
        add_message(role="user", message=user_input)
        add_message(role="assistant", message=ai_answer)
