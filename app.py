import os
import streamlit as st
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load variables from .env
load_dotenv()


# LangSmith configuration (optional - only if you have a LangSmith account)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Grok Chatbot"


# Check for required API key
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    st.error("⚠️ Missing XAI_API_KEY!\n\nPlease create a `.env` file in this directory with:\n```\nXAI_API_KEY=your_api_key_here\n```\n\nGet your API key from: https://console.x.ai")
    st.stop()


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.
        Answer questions clearly and concisely.
        If you don't know the answer, say "I don't know".
        Do not make up information."""
    ),
    (
        "user",
        "Question: {question}"
    )
])


# Streamlit app
st.title("🤖 LangChain Demo With Grok")
st.markdown("Ask me anything and I'll respond using the Grok AI model.")

input_text = st.text_input("Enter your question here:")


# Grok model
llm = ChatXAI(
    model="grok-4.5",
    xai_api_key=xai_api_key
)


# Output parser
output_parser = StrOutputParser()


# Create chain
try:
    chain = prompt | llm | output_parser
except Exception as e:
    st.error(f"Failed to create chain: {str(e)}")
    st.stop()


# Run chain when user enters a question
if input_text:
    try:
        with st.spinner("Getting response from Grok..."):
            response = chain.invoke({
                "question": input_text
            })
        st.success("Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        if "Incorrect API" in str(e):
            st.info("Please verify your XAI_API_KEY is correct in the `.env` file.")