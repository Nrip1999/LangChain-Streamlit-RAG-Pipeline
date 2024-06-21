from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
import streamlit as st


#Get API keys from .env file
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['SERPAPI_API_KEY']=os.getenv("SERPAPI_API_KEY")
JobsQuery = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

#stackexchange = StackExchangeAPIWrapper()

#Load contract to query
loader=PyPDFLoader("06._Volume_II_Section_VI_Conditions_of_Contract_Particular_Conditions.pdf")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectordb.as_retriever()

retriever_tool=create_retriever_tool(retriever,"contract_search",
                      "Search for information about Contract.")

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools=[wiki,arxiv,retriever_tool,JobsQuery]

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")

agent=create_openai_tools_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

#Web App title
st.title('Langchain RAG Demo With OPENAI API')

#Input queries in streamlit text input
query=st.text_input("Please enter query about Contract : ")
query2=st.text_input("Please enter query about jobs : ")
query3=st.text_input("Please enter Wiki query : ")
query4=st.text_input("Please enter ArXiv query : ")

#Response generator
response = agent_executor.invoke({"input":query})
response
response2 = agent_executor.invoke({"input":query2})
response2
response3 = agent_executor.invoke({"input":query3})
response3
response4 = agent_executor.invoke({"input":query4})
response4


