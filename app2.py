import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI 醫院心臟疾病問答機", page_icon="🏥")

# ---------------------------
# Sidebar: API Key inputs
# ---------------------------
st.sidebar.header("🔑 金鑰設定")

# Groq API Key: 必填（使用者輸入後才啟動 RAG）
groq_key_input = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.session_state.get("GROQ_API_KEY", ""),
    placeholder="gsk_...",
)

# (可選) HuggingFace Token：若你 embeddings / 模型需要 HF 權限才填
hf_token_input = st.sidebar.text_input(
    "Hugging Face Token (可選)",
    type="password",
    value=st.session_state.get("HUGGING_FACE_HUB_TOKEN", ""),
    placeholder="hf_...",
)

# 按鈕：明確由使用者觸發「開始」
start = st.sidebar.button("啟動 / 更新金鑰", type="primary")

if start:
    # 存到 session_state，避免每次 rerun 都要重打
    st.session_state["GROQ_API_KEY"] = groq_key_input.strip()
    st.session_state["HUGGING_FACE_HUB_TOKEN"] = hf_token_input.strip()

# 讓程式本次執行也能讀到（LangChain/Groq client 通常從參數讀，但你可能也會用 env）
if st.session_state.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"]

if st.session_state.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = st.session_state["HUGGING_FACE_HUB_TOKEN"]


# ---------------------------
# Lazy load resources (AFTER key is set)
# ---------------------------
@st.cache_resource
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_db",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource
def load_llm(groq_api_key: str):
    return ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="gemma-7b-it",
    )

# ---------------------------
# Prompt template
# ---------------------------
system_prompt = "你是心臟科的實習醫生，請根據資料來回應病患的問題。請親切、簡潔並附帶具體建議。請用台灣習慣的中文回應。"
prompt_template = """
根據下列資料：
{retrieved_chunks}

回答使用者的問題：{question}

請根據資料內容回覆，若資料不足請告訴病患可以前往最近的醫院問診。
"""


# ---------------------------
# UI
# ---------------------------
st.title("🎓 AI 醫院心臟疾病問答機")

# 沒有 key 就先擋住，避免載入/呼叫失敗
if not st.session_state.get("GROQ_API_KEY"):
    st.info("請先在左側輸入 Groq API Key，按「啟動 / 更新金鑰」後再開始問答。")
    st.stop()

# 初始化 LLM / Retriever（確保 key 已存在）
try:
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    st.error(f"向量庫載入失敗：{e}\n請確認 faiss_db 目錄存在且可讀取。")
    st.stop()

try:
    llm = load_llm(st.session_state["GROQ_API_KEY"])
except Exception as e:
    st.error(f"LLM 初始化失敗：{e}\n請確認 Groq API Key 正確。")
    st.stop()


# ---------------------------
# Chat state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入你的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs = retriever.invoke(prompt)
        retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt_template.format(
            retrieved_chunks=retrieved_chunks,
            question=prompt
        )

        # Groq 的 ChatGroq 可以直接吃字串；這裡維持你原本的做法
        response = llm.invoke(system_prompt + "\n" + final_prompt).content
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
