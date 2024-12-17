import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
import os

# 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "Your langsmith api key"
os.environ["LANGCHAIN_PROJECT"] = "창조와 진화"
st.title("창조와 진화 LLM")

with st.sidebar:
    openai_api_key = st.sidebar.text_input(
        "Input Your Openai API key",
        type="password",
    )
if prompt :=st.chat_input("무엇이 궁금하세요?"):
    with st.chat_message("user"):
        st.markdown(prompt)

if openai_api_key:
    model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini",temperature=0)

    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key, model = "text-embedding-3-small"
    )
    db = Chroma(
        persist_directory="./db/chroma/creations",embedding_function=embedding_model
    )
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=model,
    )
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content = "당신은 창조과학회의 일원으로 학생들의 질문에 대해 답해주는 선생님이기도 합니다."),
        HumanMessagePromptTemplate.from_template(
            """
            질문 : {question}
            질문에 대해서 반드시 아래의 문맥에 기반하여 답해주세요.  
            주어진 질문의 의도에 대해 명확하게 답변하고, 답변이 완료된 이후에는 주어진 문맥을 추가로 제공하여 이해를 돕습니다.
            주어진 문맥 : {context}
                       
            """
        ),
    ]
)
if not openai_api_key and prompt:
    st.sidebar.warning("Please enter your OpenAI API Key.")
    with st.chat_message("assistant"):
        st.write("OpenAI API KEY 를 입력해 주세요")
if openai_api_key and prompt:
    context = retriever.invoke(prompt)
    message = chat_template.format_messages(
        question = prompt, context=context[0].page_content
    )
    def stream_response(stream):
        for chunk in stream:
            yield chunk.content
    with st.chat_message("assistant"):
        stream = model.stream(message)
        st.write_stream(stream_response(stream))