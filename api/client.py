import requests
import streamlit as st

def openai_output(input_text):
    output=requests.post("http://localhost:8503/essay/invoke",
    json={'input':{'topic':input_text}})

    return output.json()['output']['content']

def ollama_output(input_text):
    output=requests.post(
    "http://localhost:8503/poem/invoke",
    json={'input':{'topic':input_text}})

    return output.json()['output']

    ## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Write an essay on")
input_text1=st.text_input("Write a poem on")

if input_text:
    st.write(openai_output(input_text))

if input_text1:
    st.write(ollama_output(input_text1))
