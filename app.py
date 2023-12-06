import streamlit as st
import pandas as pd
from backend.functions import *

# page setup
st.set_page_config(
    page_title="Movie Recommendation With PaLM-2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# sidebar
with st.sidebar:
    st.title("🍿 Movie Recommendation With PaLM-2")
    st.sidebar.caption("MIT License © 2023 keanteng")
    with st.expander("PaLM-2 API", expanded=True):
        api_toggle = st.toggle("Enable PaLM-2 API", value=False)
        api_input = st.text_input(
            "PaLM-2 API Token",
            type="password",
            placeholder="Enter your PaLM-2 API token here",
        )


# main page

## load data
movie_data = pd.read_excel("data.xlsx")

## configure API
if api_toggle:
    api_configure(api_key=api_input)
else:
    api_configure(api_key=AIzaSyAANEPA1UF6WE4O_0GQh2s27iBT4VrN0Ag)

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if is_valid_json(message["content"]) == True:
            temp = message["content"]
            temp = json_to_frame(message["content"])
            st.dataframe(temp, hide_index=True)
        else:
            st.markdown(message["content"])

# chatbot
prompt = st.chat_input("Enter your prompt here")

## workflow
model = load_llm()
movie_data = data_processing(movie_data)

with st.chat_message(name="AI", avatar="🎬"):
    st.write(
        "Share your thoughts on a movie you like, and I'll recommend you a movie you might like!"
    )

if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    to_llm = prompt_processing(prompt, movie_data)
    response = llm_agent(prompt=to_llm, model=model)
    response_df = json_to_frame(response)
    st.chat_message("assistant").dataframe(response_df, hide_index=True)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
