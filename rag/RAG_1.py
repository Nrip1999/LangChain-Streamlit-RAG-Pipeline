from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

#load,chunk and index the content of the PDF

loader=PyPDFLoader('Comparison_of_the_Usability_of_Apple_M2_and_M1_Pro.pdf')
doc_arr=loader.load()

Chunk_Gen=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=Chunk_Gen.split_documents(doc_arr)

db = Chroma.from_documents(documents,OpenAIEmbeddings())

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Please respond to the user queries."),
        ("user","Question:{question}")
    ]
)

st.title('Langchain RAG Demo With OPENAI API')
query=st.text_input("Please enter query")

llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

retrieved_results=db.similarity_search(query)
print(retrieved_results[0].page_content)

