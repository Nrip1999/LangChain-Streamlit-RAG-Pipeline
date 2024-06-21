from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from fastapi import FastAPI
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from pyngrok import ngrok
import nest_asyncio

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="API Server"

)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)
#Get OpenAI llm
model=ChatOpenAI()
##ollama llama2
llm=Ollama(model="llama2")

prompt1=ChatPromptTemplate.from_template("Write me a 200 word essay about {topic} which can get maximum points in SAT")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} which is good enough to be published")

add_routes(
    app,
    prompt1|model,
    path="/essay"


)

add_routes(
    app,
    prompt2|llm,
    path="/poem"


)


if __name__=="__main__":
    # Set up ngrok tunnel
    #public_url = ngrok.connect(8000)
    #print(f"Public URL: {public_url}")
    #nest_asyncio.apply()    
    #uvicorn.run(app,host="localhost",port=8000)
    uvicorn.run(app,host="127.0.0.1",port=8503)

