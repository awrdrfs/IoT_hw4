import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Set Hugging Face Hub token from Gemma_KEY
if os.getenv("Gemma_KEY"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("Gemma_KEY")


# Custom embedding class from the notebook
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

    def embed_documents(self, texts):
        # You can also change "none" to the real title (filename/chapter name) for better stability
        texts = [f"title: none | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # Official retrieval suggestion prefix
        return super().embed_query(f"task: search result | query: {text}")

# Load the vector store
@st.cache_resource
def load_vector_store():
    embedding_model = EmbeddingGemmaEmbeddings()
    vectorstore = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Set up the language model
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

# Set up the prompt template
system_prompt = "你是心臟科的實習醫生，請根據資料來回應病患的問題。請親切、簡潔並附帶具體建議。請用台灣習慣的中文回應。"
prompt_template = """
根據下列資料：
{retrieved_chunks}

回答使用者的問題：{question}

請根據資料內容回覆，若資料不足請告訴病患可以前往最近的醫院問診。
"""

# Create the Streamlit app
st.title("🎓 AI 醫院心臟疾病問答機")

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
        final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=prompt) 
        
        response = llm.invoke(system_prompt + "\n" + final_prompt).content
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})