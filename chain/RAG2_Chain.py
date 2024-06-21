import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

PDF_Loader = PyPDFLoader("Comparison_of_the_Usability_of_Apple_M2_and_M1_Pro.pdf")
docs = PDF_Loader.load()

Chunk_Gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

docs_arr=Chunk_Gen.split_documents(docs)

db=FAISS.from_documents(docs_arr[:30],OpenAIEmbeddings())

llm=ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Please give a detailed step-by-step answer.
<context>
{context}
</context>
Question: {input}""")

document_chain=create_stuff_documents_chain(llm,prompt)

retriever=db.as_retriever()

retrieval_chain=create_retrieval_chain(retriever,document_chain)

st.title('Langchain RAG Demo With OPENAI API')
query=st.text_input("Please enter query")

response=retrieval_chain.invoke({"input":query})

response['answer']