import streamlit as st
from st_chat_message import message
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

loader = DirectoryLoader("medicines", glob="*.txt", loader_cls=TextLoader)
medicines = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(medicines)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cuda"})

persist_directory = 'db'

vectordb = Chroma.from_documents(documents=text_chunks, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)


# Default System Prompt for Llama Models
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

SYSTEM_PROMPT = """
You are a helpful AI assistant who answers questions about medicines. You are not a medical professional, you are an AI who answers questions about medicines. 
If you don't know the answer to a question, please don't share false information. Use the following pieces of context to answer the question at the end. 
If you don't find the answer in the given context, just say that you don't know, don't try to make up an answer. 
Do not mention that you are using the given context, just use the given context and answer.
""".strip()

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)


prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':256, "temperature":0})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#retriever = vector_store.as_retriever(search_kwargs={"k":3})
retriever = vectordb.as_retriever(search_kwargs={"k":1})

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                              retriever= retriever,
                                              memory= memory,
                                              combine_docs_chain_kwargs={"prompt": prompt})

st.title("MedBot 💊")

def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I am MedBot. Your personal medicine AI, ask me anything you need to know about a medicine."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me anything about medicines you need!", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


initialize_session_state()
display_chat_history()