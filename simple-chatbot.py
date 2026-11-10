import utils.session
import warnings
from langchain._api import LangChainDeprecationWarning
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory import FileChatMessageHistory

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
_=load_dotenv(find_dotenv())
openai_api_key = os.environ["GROQ_API_KEY"]
chatbot = ChatGroq(model="qwen/qwen3.8-27b")

# messagesToTheChatbot = [
#     HumanMessage(content="my favorite food is pizza"),
# ]

# chatbot.invoke(messagesToTheChatbot)

# chatbot.invoke([
#     HumanMessage(content="what is my favorite food?")
# ])

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chatbot,
    prompt = prompt,
    memory=memory
)

# chain.invoke("hello!")

# chain.invoke("My name is Al Amin")
# chain.invoke("What is my name?")