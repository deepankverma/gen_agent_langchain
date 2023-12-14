from image_description.comb_tool import outputs_all_tools
from langchain.llms import OpenAI
# from dotenv import load_dotenv
# # Load environment variables from the .env file
# load_dotenv()
from langchain.agents import initialize_agent
# initialize conversational memory
# import logging
# logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta
from typing import List
# from termcolor import colored
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool  ## for custom transformer tool
from langchain.chat_models import ChatOpenAI ## for custom transformer tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory ## for custom transformer tool

from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="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")
llm = ChatOpenAI(temperature=0.9, model = "gpt-3.5-turbo")


tools = []
tools.append(outputs_all_tools())


def scene_summarizer(url):

    print(url)

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    # initialize agent with tools
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        # agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=5,
        early_stopping_method='generate',
        memory=conversational_memory,
        handle_parsing_errors=True,
    )

    sys_msg = """ Assistant is a large language model trained by OpenAI. However, assistant can't see. Therefore to help the assistant see, there are some tools which
    the assisitant can use. These tools give the understanding of the overall scene in the urban streets with the help of values. After receiving these values, assistant 
    then decides to produce a small paragraph which provides the information so a common person can understand.
    """

    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )

    agent.agent.llm_chain.prompt = new_prompt
    # update the agent tools
    agent.tools = tools

    
# img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    output= agent(f"Write a small paragraph on the visual elements you see in the street view.\n{url}")
    
    return output 

